/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header) ...
*/

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string> // Added for std::string
#include <utility>
#include <vector>

// Includes needed for types used in this header
#include "chess/types.h"     // Defines lczero::Value, lczero::GameResult, lczero::kValueMate, lczero::Move etc.
#include "chess/position.h" // Defines lczero::PositionHash (Although not directly used, might be implicitly needed or used in .cc)
#include "utils/mutex.h"    // Defines Mutex, SharedMutex etc. (If needed by impl, though not directly here)

// Forward declarations to break potential cycles and reduce header dependencies
namespace lczero {
class MoveList; // Forward declare MoveList if only pointers/references are used in the header
class GameState;
class PositionHistory;
class NodeTree; // Forward declare NodeTree
namespace classic {
class SearchParams;
class Node;
class EdgeAndNode;
template <bool is_const> class Edge_Iterator;
template <bool is_const> class VisitedNode_Iterator;
} // namespace classic
} // namespace lczero


namespace lczero {
namespace classic {

// --- Edge ---
class Edge {
 public:
  static std::unique_ptr<Edge[]> FromMovelist(const lczero::MoveList& moves); // Use lczero::
  lczero::Move GetMove(bool as_opponent = false) const; // Use lczero::
  float GetP() const;
  void SetP(float val);
  std::string DebugString() const;

 private:
  lczero::Move move_;
  uint16_t p_ = 0; // Represents policy probability (scaled)
  friend class Node;
  template <bool is_const> friend class Edge_Iterator; // Friend declaration needs template syntax
};

// --- Node ---
class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;
  using VisitedIterator = VisitedNode_Iterator<false>;
  using ConstVisitedIterator = VisitedNode_Iterator<true>;


  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // Constructor taking parent and index
  Node(Node* parent, uint16_t index);

  // Move constructor/assignment (default is okay as members handle ownership)
  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  // Disable copy constructor/assignment
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  ~Node(); // Destructor declaration

  Node* CreateSingleChildNode(lczero::Move m); // Use lczero::
  void CreateEdges(const lczero::MoveList& moves); // Use lczero::
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;

  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetVisitCount() const { return GetN(); } // Alias for GetN
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetChildrenVisits() const {
      uint32_t n = GetN();
      return n > 0 ? n - 1 : 0; // Total N - 1 (since N includes the visit that created the children)
  }
  int GetEffectiveVisits() const { return GetNStarted(); } // Alias for GetNStarted
  int GetEffectiveParentVisits() const;

  float GetQ(float draw_score) const;
  lczero::Value GetValue() const; // Use lczero::Value
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); } // Win-Loss value [-1, 1]
  float GetD() const { return d_.load(std::memory_order_relaxed); } // Draw probability [0, 1]
  float GetM() const { return m_.load(std::memory_order_relaxed); } // Moves Left estimate
  float GetPolicyPrior() const; // Policy prior of the edge leading to this node

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<lczero::GameResult, lczero::GameResult> Bounds; // Use lczero::GameResult
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  void SetBounds(lczero::GameResult lower, lczero::GameResult upper); // Use lczero::GameResult
  void MakeTerminal(lczero::GameResult result, float plies_left = 0.0f, // Use lczero::GameResult
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();

  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;

  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(lczero::Value v, float d, float m, int multivisit); // Use lczero::Value
  void AdjustForTerminal(lczero::Value v_delta, float d_delta, float m_delta, int multivisit); // Use lczero::Value
  void RevertTerminalVisits(lczero::Value v, float d, float m, int multivisit); // Use lczero::Value
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); }

  ConstIterator Edges() const;
  Iterator Edges();
  ConstVisitedIterator VisitedNodes() const; // Corrected type
  VisitedIterator VisitedNodes(); // Corrected type

  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node_to_save);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const; // Get the edge in the parent that points to this node
  std::string DebugString() const;
  bool MakeSolid(); // Marks children as solid (won't be pruned easily)
  void SortEdges(); // Sorts edges based on policy

  uint16_t Index() const { return index_; } // Index of this node within its parent's children array

   // --- NEW MEMBERS for Proven State ---
   std::atomic<bool> is_known_win{false};
   std::atomic<bool> is_known_loss{false};

   // --- NEW METHODS for Proven State ---
   lczero::Value GetMinValue() const; // Use lczero::Value
   lczero::Value GetMaxValue() const; // Use lczero::Value

   // --- Helper Methods ---
   // Note: Accessing children directly via pointers might be unsafe if nodes are moved.
   // Prefer using iterators or GetChild helper if needed.
   // const std::unique_ptr<Node>* GetChildrenPtr() const { return &child_; } // Potentially unsafe if node structure changes
   Node* GetChild(int index) const; // Safer way to access a specific child by index
   // int GetNumChildren() const { return num_edges_; } // Use GetNumEdges() instead

 private:
  void UpdateChildrenParents(); // Helper to update parent pointers when nodes are moved/restructured

  std::unique_ptr<Edge[]> edges_; // Array of edges to potential children
  Node* parent_ = nullptr; // Pointer to the parent node
  std::unique_ptr<Node> child_; // Pointer to the first child (linked list structure)
  std::unique_ptr<Node> sibling_; // Pointer to the next sibling (linked list structure)
  std::atomic<double> wl_{0.0}; // Accumulated WDL value (Win-Loss component)

  std::atomic<float> d_{0.0}; // Accumulated Draw probability
  std::atomic<float> m_{0.0}; // Accumulated Moves Left estimate
  std::atomic<uint32_t> n_{0}; // Number of completed visits
  std::atomic<uint32_t> n_in_flight_{0}; // Number of visits currently in flight (evaluation pending)

  uint16_t index_; // Index of this node in the parent's child list

  uint8_t num_edges_ = 0; // Number of legal moves (edges) from this node

  // Packed bitfield members
  Terminal terminal_type_ : 2 = Terminal::NonTerminal; // Type of terminal state (if any)
  lczero::GameResult lower_bound_ : 2 = lczero::GameResult::BLACK_WON; // Lower bound on node value
  lczero::GameResult upper_bound_ : 2 = lczero::GameResult::WHITE_WON; // Upper bound on node value
  bool solid_children_ : 1 = false; // Whether children are considered 'solid'

  // Friend declarations for classes that need internal access
  friend class lczero::NodeTree; // Needs access to manage tree structure
  template <bool is_const> friend class Edge_Iterator;
  template <bool is_const> friend class VisitedNode_Iterator;
  friend class SearchWorker; // Needs access for search operations
};


// --- EdgeAndNode ---
// Helper class combining an Edge and its corresponding Node (if visited)
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; node_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
   bool operator==(const EdgeAndNode& other) const {
       return edge_ == other.edge_ && node_ == other.node_;
   }
   bool operator!=(const EdgeAndNode& other) const { return !(*this == other); }
   bool HasNode() const { return node_ != nullptr; }
   Edge* edge() const { return edge_; }
   Node* node() const { return node_; }

   // Accessor methods that safely handle null node_ pointers
   float GetQ(float default_q, float draw_score) const; // Calculated Q-value
   float GetWL(float default_wl) const; // Win-Loss value from node or default
   float GetD(float default_d) const;   // Draw probability from node or default
   float GetM(float default_m) const;   // Moves Left from node or default
   uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
   int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
   uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
   bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
   bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
   Node::Bounds GetBounds() const; // Return type uses qualified GameResult
   float GetP() const { return edge_ ? edge_->GetP() : 0.0f; }
   lczero::Move GetMove(bool flip = false) const; // Use lczero::Move
   float GetU(float puct_numerator) const; // Calculated U-value (exploration term)
   std::string DebugString() const;

 protected:
   Edge* edge_ = nullptr;
   Node* node_ = nullptr;
};

// --- Edge_Iterator ---
// Iterator for traversing all potential edges/children of a node
template <bool is_const>
class Edge_Iterator : public EdgeAndNode { // Inherits from EdgeAndNode to provide accessors
 public:
  // Standard iterator typedefs
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using value_type = EdgeAndNode; // Dereferencing yields an EdgeAndNode view
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = const EdgeAndNode*; // Pointer to the view
  using reference = const EdgeAndNode&; // Reference to the view

  Edge_Iterator(); // Default constructor for end iterator
  Edge_Iterator(NodePtr parent_node); // Constructor for begin iterator

  // Iterator comparison
  bool operator==(const Edge_Iterator<is_const>& other) const;
  bool operator!=(const Edge_Iterator<is_const>& other) const { return !(*this == other); }

  // Iterator operations
  Edge_Iterator<is_const>& begin(); // Returns *this
  Edge_Iterator<is_const> end();   // Returns default-constructed iterator
  Edge_Iterator<is_const>& operator++(); // Pre-increment
  reference operator*() const; // Dereference returns EdgeAndNode view
  pointer operator->() const;  // Member access returns pointer to EdgeAndNode view

  // Specific method for search worker
  Node* GetOrSpawnNode(Node* parent);

 private:
  void Actualize(); // Update internal pointers based on current state

  NodePtr parent_node_ = nullptr; // The node whose edges are being iterated
  NodePtr current_node_ptr_ = nullptr; // Pointer to the current child node in the linked list
  uint16_t current_idx_ = 0; // Index corresponding to the current edge
};

// --- VisitedNode_Iterator ---
// Iterator specifically for traversing *visited* children (nodes that exist)
template <bool is_const>
class VisitedNode_Iterator {
 public:
  // Standard iterator typedefs
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using value_type = NodePtr; // Dereferencing yields a Node pointer
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = NodePtr*;
  using reference = NodePtr&;


  VisitedNode_Iterator(); // Default constructor for end iterator
  VisitedNode_Iterator(NodePtr parent_node); // Constructor for begin iterator

  // Iterator comparison
  bool operator==(const VisitedNode_Iterator<is_const>& other) const;
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return !(*this == other); }

  // Iterator operations
  VisitedNode_Iterator<is_const>& begin(); // Returns *this
  VisitedNode_Iterator<is_const> end(); // Returns default-constructed iterator
  VisitedNode_Iterator<is_const>& operator++(); // Pre-increment
  NodePtr operator*() const; // Dereference returns pointer to the visited Node

 private:
   void AdvanceToNextVisited(); // Helper to find the next valid node

   NodePtr parent_node_ = nullptr; // The node whose children are being iterated
   NodePtr current_node_ptr_ = nullptr; // Pointer to the current child node in the linked list
   // No index needed as we follow the linked list directly
};


// Inline definitions for Node iterator accessors
inline Node::ConstIterator Node::Edges() const { return ConstIterator(this); }
inline Node::Iterator Node::Edges() { return Iterator(this); }
inline Node::ConstVisitedIterator Node::VisitedNodes() const { return ConstVisitedIterator(this); }
inline Node::VisitedIterator Node::VisitedNodes() { return VisitedIterator(this); }

// Inline definitions for Edge_Iterator
template <bool is_const> inline Edge_Iterator<is_const>& Edge_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline Edge_Iterator<is_const> Edge_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline typename Edge_Iterator<is_const>::reference Edge_Iterator<is_const>::operator*() const { return *this; } // Return reference to self (as EdgeAndNode view)
template <bool is_const> inline typename Edge_Iterator<is_const>::pointer Edge_Iterator<is_const>::operator->() const { return this; } // Return pointer to self

// Inline definitions for VisitedNode_Iterator
template <bool is_const> inline VisitedNode_Iterator<is_const>& VisitedNode_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline VisitedNode_Iterator<is_const> VisitedNode_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline typename VisitedNode_Iterator<is_const>::NodePtr VisitedNode_Iterator<is_const>::operator*() const { return current_node_ptr_; }


}  // namespace classic
}  // namespace lczero
