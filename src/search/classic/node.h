/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <algorithm>
#include <atomic>       // For std::atomic
#include <cmath>
#include <iostream>     // For std::ostream (used in DebugString potentially)
#include <limits>       // For std::numeric_limits
#include <memory>       // For std::unique_ptr
#include <mutex>        // For std::mutex (used in utils/mutex.h)
#include <string>       // For std::string
#include <utility>      // For std::pair, std::move
#include <vector>       // For std::vector (used potentially in impl or callers)
#include <type_traits>  // For std::conditional_t

// Includes needed for types used in this header
#include "chess/types.h"     // Defines lczero::Value, lczero::GameResult, lczero::kValueMate, lczero::Move, lczero::MoveList etc.
#include "chess/position.h" // Defines lczero::PositionHash (if needed, though not directly used here)
#include "utils/mutex.h"    // Defines Mutex, SharedMutex etc. (if needed by impl)


namespace lczero {

// Forward declarations from other namespaces/files
class NodeTree;
class Position;
class PositionHistory;
class UciResponder;
class SyzygyTablebase;
class Backend;
class BackendComputation;
class BackendAttributes;
class OptionsDict;
class SearchStopper;
class StoppersHints;
class IterationStats;
struct EvalResult; // Defined in proto/net.pb.h, forward declare ok if only pointer/ref used in header

namespace classic {

// Forward declarations within this namespace
class SearchParams;
class Node;
class Edge;
class EdgeAndNode;
template <bool is_const> class Edge_Iterator;
template <bool is_const> class VisitedNode_Iterator;
class SearchWorker;


// --- Edge ---
// Represents a potential move from a parent Node.
class Edge {
 public:
  // Creates an array of edges from a list of legal moves.
  static std::unique_ptr<Edge[]> FromMovelist(const lczero::MoveList& moves);

  // Returns the move associated with this edge.
  // `as_opponent` flag is likely deprecated or unused, standard move representation should suffice.
  lczero::Move GetMove(bool as_opponent = false) const;

  // Returns the policy prior probability for this move [0, 1].
  float GetP() const;
  // Sets the policy prior probability for this move.
  void SetP(float val);

  // Returns a debug string representation of the edge.
  std::string DebugString() const;

 private:
  lczero::Move move_; // The chess move this edge represents.
  uint16_t p_ = 0;   // Policy probability stored in compressed float16 format.

  // Allow Node and iterators access to private members.
  friend class Node;
  template <bool is_const> friend class Edge_Iterator;
};


// --- Node ---
// Represents a position in the search tree.
class Node {
 public:
  // --- Typedefs ---
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;
  using VisitedIterator = VisitedNode_Iterator<false>;
  using ConstVisitedIterator = VisitedNode_Iterator<true>;

  // --- Enum for Terminal State ---
  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // --- Constructor ---
  // Takes pointer to a parent node and own index within the parent's edges.
  Node(Node* parent, uint16_t index);

  // --- Destructor ---
  ~Node(); // Custom destructor needed due to solid_children hack

  // --- Move Semantics ---
  // Default move constructor/assignment are okay as members manage ownership.
  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  // --- Copy Semantics ---
  // Nodes represent unique states in the tree, copying is disallowed.
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  // --- Tree Structure Management ---
  Node* CreateSingleChildNode(lczero::Move m);
  void CreateEdges(const lczero::MoveList& moves);
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node_to_save);
  bool MakeSolid(); // Marks children as solid (stored contiguously)

  // --- Visit and Value Accessors ---
  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetChildrenVisits() const { uint32_t n = GetN(); return n > 0 ? n - 1 : 0; }
  int GetEffectiveVisits() const { return GetNStarted(); } // Alias
  int GetEffectiveParentVisits() const;

  lczero::Value GetValue() const; // Returns averaged value (Win probability - Loss probability) [-1, 1]
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); } // W-L value
  float GetD() const { return d_.load(std::memory_order_relaxed); }   // Draw probability
  float GetM() const { return m_.load(std::memory_order_relaxed); }   // Moves Left estimate
  float GetQ(float draw_score) const; // Q-value incorporating draw score

  // --- Policy Accessors ---
  float GetVisitedPolicy() const; // Sum of policy priors of visited children
  float GetPolicyPrior() const; // Policy prior of the edge leading *to* this node
  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;
  void SortEdges(); // Sorts edges based on policy prior

  // --- Terminal State Management ---
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  void MakeTerminal(lczero::GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();

  // --- Bounds Management ---
  typedef std::pair<lczero::GameResult, lczero::GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  void SetBounds(lczero::GameResult lower, lczero::GameResult upper);
  lczero::Value GetMinValue() const; // Returns lower bound as Value [-1, 1]
  lczero::Value GetMaxValue() const; // Returns upper bound as Value [-1, 1]

  // --- Score Update Management (for SearchWorker) ---
  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(lczero::Value v, float d, float m, int multivisit);
  void AdjustForTerminal(lczero::Value v_delta, float d_delta, float m_delta, int multivisit);
  void RevertTerminalVisits(lczero::Value v, float d, float m, int multivisit);
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); }

  // --- Iterators ---
  ConstIterator Edges() const;
  Iterator Edges();
  ConstVisitedIterator VisitedNodes() const;
  VisitedIterator VisitedNodes();

  // --- Helpers ---
  Edge* GetEdgeToNode(const Node* node) const; // Finds edge corresponding to a child node
  Edge* GetOwnEdge() const; // Gets the edge in the parent pointing to this node
  uint16_t Index() const { return index_; } // Index of this node in parent's edge list
  Node* GetChild(int index) const; // Gets child node at a specific edge index (if exists)
  std::string DebugString() const;

  // --- Proven State Flags ---
  std::atomic<bool> is_known_win{false};  // True if subtree guarantees a win
  std::atomic<bool> is_known_loss{false}; // True if subtree guarantees a loss

 private:
  // --- Private Methods ---
  void UpdateChildrenParents(); // Updates parent pointers after node moves

  // --- Member Variables (arranged roughly by size for packing) ---

  // 8-byte fields
  std::unique_ptr<Edge[]> edges_;         // Array of edges to potential children
  Node* parent_ = nullptr;                // Pointer to the parent node
  std::unique_ptr<Node> child_;           // Pointer to the first child (linked list or array if solid)
  std::unique_ptr<Node> sibling_;         // Pointer to the next sibling (linked list)
  std::atomic<double> wl_{0.0};           // Accumulated WDL value (W-L)

  // 4-byte fields
  std::atomic<float> d_{0.0};             // Accumulated Draw probability
  std::atomic<float> m_{0.0};             // Accumulated Moves Left estimate
  std::atomic<uint32_t> n_{0};            // Number of completed visits
  std::atomic<uint32_t> n_in_flight_{0};  // Number of visits in flight

  // 2-byte fields
  uint16_t index_;                       // Index of this node in parent's edge list

  // 1-byte fields
  uint8_t num_edges_ = 0;                // Number of legal moves (edges)

  // Bit fields (packed into 1 byte)
  Terminal terminal_type_ : 2;           // Type of terminal state
  lczero::GameResult lower_bound_ : 2;   // Lower bound on node value
  lczero::GameResult upper_bound_ : 2;   // Upper bound on node value
  bool solid_children_ : 1;              // Whether children are stored contiguously

  // --- Friend Declarations ---
  friend class lczero::NodeTree;
  template <bool is_const> friend class Edge_Iterator;
  template <bool is_const> friend class VisitedNode_Iterator;
  friend class SearchWorker;
};


// --- EdgeAndNode ---
// Helper class combining an Edge and its corresponding Node (if visited).
// Provides convenient accessors.
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; node_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; } // True if edge exists
  bool operator==(const EdgeAndNode& other) const { return edge_ == other.edge_ && node_ == other.node_; }
  bool operator!=(const EdgeAndNode& other) const { return !(*this == other); }

  // Basic accessors
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  // Accessors that safely handle null node_ pointers
  float GetQ(float default_q, float draw_score) const;
  float GetWL(float default_wl) const;
  float GetD(float default_d) const;
  float GetM(float default_m) const;
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Node::Bounds GetBounds() const;
  float GetP() const { return edge_ ? edge_->GetP() : 0.0f; }
  lczero::Move GetMove(bool flip = false) const;
  float GetU(float puct_numerator) const; // Calculated U-value (PUCT exploration term)
  std::string DebugString() const;

 protected:
  Edge* edge_ = nullptr; // Pointer to the Edge object
  Node* node_ = nullptr; // Pointer to the corresponding Node (null if not visited/spawned)
};


// --- Edge_Iterator ---
// Iterator for traversing all potential edges/children of a node.
// Dereferencing yields an EdgeAndNode view.
template <bool is_const>
class Edge_Iterator : public EdgeAndNode { // Inherits to provide the view directly
 public:
  // Standard iterator typedefs
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*, std::unique_ptr<Node>*>; // Pointer to unique_ptr for linked list manipulation
  using value_type = EdgeAndNode; // What operator* returns (conceptually)
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = const EdgeAndNode*; // Pointer to the iterator itself (acting as the value)
  using reference = const EdgeAndNode&; // Reference to the iterator itself (acting as the value)

  Edge_Iterator(); // Default constructor -> end() iterator
  Edge_Iterator(NodePtr parent_node); // Constructor for begin() iterator

  // Iterator comparison
  bool operator==(const Edge_Iterator<is_const>& other) const;
  bool operator!=(const Edge_Iterator<is_const>& other) const { return !(*this == other); }

  // Iterator operations
  Edge_Iterator<is_const>& begin(); // Returns *this
  Edge_Iterator<is_const> end();   // Returns default-constructed iterator
  Edge_Iterator<is_const>& operator++(); // Pre-increment
  reference operator*() const; // Returns reference to *this (the EdgeAndNode view)
  pointer operator->() const; // Returns pointer to *this

  // Specific method needed by SearchWorker to potentially create the node
  Node* GetOrSpawnNode(Node* parent);

 private:
  void Actualize(); // Updates edge_ and node_ based on current iterator state

  NodePtr parent_node_ = nullptr;       // The node whose edges we are iterating
  Ptr node_ptr_ = nullptr;              // Pointer-to-unique_ptr for traversing/modifying the child linked list (or null if solid)
  uint16_t current_idx_ = 0;            // Current index into the edges_ array
};


// --- VisitedNode_Iterator ---
// Iterator specifically for traversing *visited* children (nodes that exist).
// Dereferencing yields a pointer to the visited Node.
template <bool is_const>
class VisitedNode_Iterator {
 public:
  // Standard iterator typedefs
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using value_type = NodePtr; // Dereferencing yields a Node pointer
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = const NodePtr*; // Pointer to the NodePtr
  using reference = const NodePtr&; // Reference to the NodePtr

  VisitedNode_Iterator(); // Default constructor -> end() iterator
  VisitedNode_Iterator(NodePtr parent_node); // Constructor for begin() iterator

  // Iterator comparison
  bool operator==(const VisitedNode_Iterator<is_const>& other) const;
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return !(*this == other); }

  // Iterator operations
  VisitedNode_Iterator<is_const>& begin(); // Returns *this
  VisitedNode_Iterator<is_const> end(); // Returns default-constructed iterator
  VisitedNode_Iterator<is_const>& operator++(); // Pre-increment
  NodePtr operator*() const; // Returns pointer to the current visited Node

 private:
   void AdvanceToNextVisited(); // Helper to find the next valid node

   NodePtr parent_node_ = nullptr;       // Node whose children are being iterated
   NodePtr current_node_ptr_ = nullptr;  // Pointer to the current visited node
   uint16_t current_idx_ = 0;            // Current index (only used if solid)
};


// --- Inline Implementations ---

// Node iterator accessors
inline Node::ConstIterator Node::Edges() const { return ConstIterator(this); }
inline Node::Iterator Node::Edges() { return Iterator(this); }
inline Node::ConstVisitedIterator Node::VisitedNodes() const { return ConstVisitedIterator(this); }
inline Node::VisitedIterator Node::VisitedNodes() { return VisitedIterator(this); }

// Edge_Iterator methods
template <bool is_const> inline Edge_Iterator<is_const>& Edge_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline Edge_Iterator<is_const> Edge_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline typename Edge_Iterator<is_const>::reference Edge_Iterator<is_const>::operator*() const { return *this; }
template <bool is_const> inline typename Edge_Iterator<is_const>::pointer Edge_Iterator<is_const>::operator->() const { return this; }
template <bool is_const> inline bool Edge_Iterator<is_const>::operator==(const Edge_Iterator<is_const>& other) const { return edge_ == other.edge_; } // Compare based on edge pointer

// VisitedNode_Iterator methods
template <bool is_const> inline VisitedNode_Iterator<is_const>& VisitedNode_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline VisitedNode_Iterator<is_const> VisitedNode_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline typename VisitedNode_Iterator<is_const>::NodePtr VisitedNode_Iterator<is_const>::operator*() const { return current_node_ptr_; }
template <bool is_const> inline bool VisitedNode_Iterator<is_const>::operator==(const VisitedNode_Iterator<is_const>& other) const { return current_node_ptr_ == other.current_node_ptr_; } // Compare based on node pointer

// Node Constructor Implementation (needed here due to bitfield init)
inline Node::Node(Node* parent, uint16_t index)
    : parent_(parent),
      index_(index),
      // Explicitly initialize bitfield members
      terminal_type_(Terminal::NonTerminal),
      lower_bound_(lczero::GameResult::BLACK_WON),
      upper_bound_(lczero::GameResult::WHITE_WON),
      solid_children_(false) {}


// Size assertion (adjust values if members change)
#if defined(_M_IX86) || defined(__i386__) || (defined(__arm__) && !defined(__aarch64__))
static_assert(sizeof(Node) == 48 || sizeof(Node) == 52, "Unexpected size of Node for 32bit compile"); // Allow some flexibility
#else
static_assert(sizeof(Node) == 64, "Unexpected size of Node");
#endif


}  // namespace classic
}  // namespace lczero
