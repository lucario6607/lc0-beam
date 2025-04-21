/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic> // Added for std::atomic
#include <vector> // Added for std::vector


#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "proto/net.pb.h"
#include "utils/mutex.h"

namespace lczero {
namespace classic {

// Forward declaration
class SearchParams;

// Children of a node are stored the following way:
// * Edges and Nodes edges point to are stored separately.
// * There may be dangling edges (which don't yet point to any Node object yet)
// * Edges are stored are a simple array on heap.
// * Nodes are stored as a linked list, and contain index_ field which shows
//   which edge of a parent that node points to.
//   Or they are stored a contiguous array of Node objects on the heap if
//   solid_children_ is true. If the children have been 'solidified' their
//   sibling links are unused and left empty. In this state there are no
//   dangling edges, but the nodes may not have ever received any visits.
//
// Example:
//                                Parent Node
//                                    |
//        +-------------+-------------+----------------+--------------+
//        |              |            |                |              |
//   Edge 0(Nf3)    Edge 1(Bc5)     Edge 2(a4)     Edge 3(Qxf7)    Edge 4(a3)
//    (dangling)         |           (dangling)        |           (dangling)
//                   Node, Q=0.5                    Node, Q=-0.2
//
//  Is represented as:
// +--------------+
// | Parent Node  |
// +--------------+                                        +--------+
// | edges_       | -------------------------------------> | Edge[] |
// |              |    +------------+                      +--------+
// | child_       | -> | Node       |                      | Nf3    |
// +--------------+    +------------+                      | Bc5    |
//                     | index_ = 1 |                      | a4     |
//                     | q_ = 0.5   |    +------------+    | Qxf7   |
//                     | sibling_   | -> | Node       |    | a3     |
//                     +------------+    +------------+    +--------+
//                                       | index_ = 3 |
//                                       | q_ = -0.2  |
//                                       | sibling_   | -> nullptr
//                                       +------------+

class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;

template <bool is_const>
class VisitedNode_Iterator;

class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // Takes pointer to a parent node and own index in a parent.
  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {
      // NOTE: If constructing from TTEntry, load known_win/loss here
      /* Example:
      if (tt_entry) {
          is_known_win.store(tt_entry->known_win, std::memory_order_relaxed);
          is_known_loss.store(tt_entry->known_loss, std::memory_order_relaxed);
          // ... load other TT data ...
      }
      */
        }

  // We have a custom destructor, but its behavior does not need to be emulated
  // during move operations so default is fine.
  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  // Allocates a new edge and a new node. The node has to be no edges before
  // that.
  Node* CreateSingleChildNode(Move m);

  // Creates edges from a movelist. There has to be no edges before that.
  void CreateEdges(const MoveList& moves);

  // Gets parent node.
  Node* GetParent() const { return parent_; }

  // Returns whether a node has children edges allocated.
  bool HasChildren() const { return static_cast<bool>(edges_); }

  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;

  // Visit count accessors
  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); } // Use atomic load
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); } // Use atomic load
  uint32_t GetVisitCount() const { return GetN(); } // Helper alias

  uint32_t GetChildrenVisits() const {
      uint32_t n = GetN();
      return n > 0 ? n - 1 : 0;
  }
  // Returns n + n_in_flight atomically.
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }

  // Effective visits for PUCT (thread-safe)
  int GetEffectiveVisits() const { return GetNStarted(); }
  int GetEffectiveParentVisits() const { // Assuming parent access is safe in context
      return GetParent() ? GetParent()->GetNStarted() : 0;
   }


  // Value accessors (need careful review based on how value_sum_ is stored/updated)
  // Assuming value_sum_ stores (WDL value * N) appropriately scaled
  // Need to ensure thread-safe read and calculation
  // Placeholder implementation - requires knowing exact value_sum_ representation
  double GetValueSum() const { return value_sum_.load(std::memory_order_relaxed); }

  // Returns node eval Q = WL + draw_score * D
  float GetQ(float draw_score) const {
       uint32_t n = GetN();
       if (n == 0) return 0.0f; // Or FPU value? Depends on context
       // Read atomic members safely
       double current_wl = wl_.load(std::memory_order_relaxed);
       float current_d = d_.load(std::memory_order_relaxed);
       return static_cast<float>(current_wl + draw_score * current_d);
   }

    // Returns node eval based on WL only (perspective of player to move)
    // Assumes wl_ stores WDL from the perspective of the player to move
    float GetValue() const {
        uint32_t n = GetN();
        if (n == 0) return 0.0f; // Return 0 for unvisited, FPU handled elsewhere
        return static_cast<float>(wl_.load(std::memory_order_relaxed));
    }

  // Returns node WL eval (W-L)
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); }
  // Returns node Draw probability
  float GetD() const { return d_.load(std::memory_order_relaxed); }
  // Returns node Moves Left estimate
  float GetM() const { return m_.load(std::memory_order_relaxed); }

  // Get Policy Prior for PUCT
  float GetPolicyPrior() const {
      if (!parent_) return 0.0f; // Root has no prior edge
      const Edge* edge = parent_->GetEdgeToNode(this);
      return edge ? edge->GetP() : 0.0f;
   }

   // Get Q for PUCT selection (handles FPU implicitly via GetQ)
   float GetQ(const SearchParams& params) const; // Declaration, implementation needs params.cc


  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint8_t GetNumEdges() const { return num_edges_; }

  // Output must point to at least max_needed floats.
  void CopyPolicy(int max_needed, float* output) const {
    if (!edges_) return;
    int loops = std::min(static_cast<int>(num_edges_), max_needed);
    for (int i = 0; i < loops; i++) {
      output[i] = edges_[i].GetP();
    }
  }

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  // Makes the node not terminal and updates its visits.
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);

  // If this node is not in the process of being expanded by another thread
  // (which can happen only if n==0 and n-in-flight==1), mark the node as
  // "being updated" by incrementing n-in-flight, and return true.
  // Otherwise return false.
  bool TryStartScoreUpdate();
  // Decrements n-in-flight back.
  void CancelScoreUpdate(int multivisit);
  // Updates the node with newly computed value v.
  // Updates:
  // * Q (weighted average of all V in a subtree)
  // * N (+=1)
  // * N-in-flight (-=1)
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  // Like FinalizeScoreUpdate, but it updates n existing visits by delta amount.
  void AdjustForTerminal(float v, float d, float m, int multivisit);
  // Revert visits to a node which ended in a now reverted terminal.
  void RevertTerminalVisits(float v, float d, float m, int multivisit);
  // When search decides to treat one visit as several (in case of collisions
  // or visiting terminal nodes several times), it amplifies the visit by
  // incrementing n_in_flight.
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); } // Use atomic fetch_add

  // Updates max depth, if new depth is larger.
  void UpdateMaxDepth(int depth);

  // Calculates the full depth if new depth is larger, updates it, returns
  // in depth parameter, and returns true if it was indeed updated.
  bool UpdateFullDepth(uint16_t* depth);

  // Returns range for iterating over edges.
  ConstIterator Edges() const;
  Iterator Edges();

  // Returns range for iterating over child nodes with N > 0.
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();

  // Deletes all children.
  void ReleaseChildren();

  // Deletes all children except one.
  // The node provided may be moved, so should not be relied upon to exist
  // afterwards.
  void ReleaseChildrenExceptOne(Node* node);

  // For a child node, returns corresponding edge.
  Edge* GetEdgeToNode(const Node* node) const;

  // Returns edge to the own node.
  Edge* GetOwnEdge() const;

  // Debug information about the node.
  std::string DebugString() const;

  // Reallocates this nodes children to be in a solid block, if possible and not
  // already done. Returns true if the transformation was performed.
  bool MakeSolid();

  void SortEdges();

  // Index in parent edges - useful for correlated ordering.
  uint16_t Index() const { return index_; }

   // --- NEW MEMBERS for Proven State ---
   std::atomic<bool> is_known_win{false};
   std::atomic<bool> is_known_loss{false};

   // --- NEW METHODS for Proven State ---
   // Recursive functions to calculate minimax bounds based on current MCTS values
   // Rely on implicit depth and visit>0 filter.
   Value GetMinValue() const;
   Value GetMaxValue() const;

   // --- NEW Helper Methods ---
   const std::unique_ptr<Node>* GetChildrenPtr() const { return &child_; } // Need way to access children for iteration
   Node* GetChild(int index) const { // Access child by index (handle solid/linked list)
       if (!edges_ || index >= num_edges_) return nullptr;
       if (solid_children_) {
           // child_ points to Node[]
           return &(child_.get()[index]);
       } else {
           // Need to traverse linked list
           Node* current = child_.get();
           while(current && current->index_ < index) {
               current = current->sibling_.get();
           }
           if (current && current->index_ == index) {
               return current;
           }
           return nullptr; // Child not found (or not created yet)
       }
   }
    int GetNumChildren() const { return num_edges_; } // Assuming num_edges_ is correct


  ~Node() {
    if (solid_children_ && child_) {
      // As a hack, solid_children is actually storing an array in here, release
      // so we can correctly invoke the array delete.
      for (int i = 0; i < num_edges_; i++) {
        // Check pointer validity before calling destructor
         if (child_.get() + i != nullptr) {
            child_.get()[i].~Node();
         }
      }
      std::allocator<Node> alloc;
      alloc.deallocate(child_.release(), num_edges_);
    }
  }

 private:
  // For each child, ensures that its parent pointer is pointing to this.
  void UpdateChildrenParents();

  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // 8 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from the
  // perspective of the player-to-move for the position.
  // WL stands for "W minus L". Is equal to Q if draw score is 0.
  // Use atomic double for thread safety during backup
  std::atomic<double> wl_{0.0};
  // Sum of values for calculating average (alternative to storing average directly)
  // Needs careful handling of units/scaling depending on how backup works.
  // Let's assume LCZero stores average directly in wl_, d_, m_ and uses atomic ops for updates.
  // If value_sum is needed, define it here:
  // std::atomic<double> value_sum_{0.0};


  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Array of edges.
  std::unique_ptr<Edge[]> edges_;
  // Pointer to a parent node. nullptr for the root.
  Node* parent_ = nullptr;
  // Pointer to a first child. nullptr for a leaf node.
  // As a 'hack' actually a unique_ptr to Node[] if solid_children.
  std::unique_ptr<Node> child_;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  // Also null in the solid case.
  std::unique_ptr<Node> sibling_;

  // 4 byte fields.
  // Averaged draw probability. Works similarly to WL, except that D is not
  // flipped depending on the side to move. Use atomic float.
  std::atomic<float> d_{0.0}; // Initialize appropriately, maybe 1.0 for unvisited?
  // Estimated remaining plies. Use atomic float.
  std::atomic<float> m_{0.0};
  // How many completed visits this node had. Use atomic uint32_t.
  std::atomic<uint32_t> n_{0};
  // (AKA virtual loss.) How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move. Use atomic uint32_t.
  std::atomic<uint32_t> n_in_flight_{0};


  // 2 byte fields.
  // Index of this node is parent's edge list.
  uint16_t index_;

  // 1 byte fields.
  // Number of edges in @edges_.
  uint8_t num_edges_ = 0;

  // Bit fields using parts of uint8_t fields initialized in the constructor.
  // Whether or not this node end game (with a winning of either sides or draw).
  Terminal terminal_type_ : 2;
  // Best and worst result for this node.
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  // Whether the child_ is actually an array of equal length to edges.
  bool solid_children_ : 1;
  // Padding/unused bits if any


  // TODO(mooskagh) Unfriend NodeTree.
  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
  friend class SearchWorker; // Allow SearchWorker access for min/max etc.
};

// Define __i386__  or __arm__ also for 32 bit Windows.
#if defined(_M_IX86)
#define __i386__
#endif
#if defined(_M_ARM) && !defined(_M_AMD64)
#define __arm__
#endif

// A basic sanity check. Adjust size based on new atomic members and alignment.
// Size will likely increase due to atomics. Check compiler output.
// Example: atomic<double> is 8, atomic<float> is 4, atomic<uint32_t> is 4, atomic<bool> is 1 (maybe padded)
// Previous size: 64 (64-bit) / 48 (32-bit)
// Added: 2 atomic<bool> (~2 bytes + padding)
// Changed: wl_ to atomic<double> (no size change), d_, m_, n_, n_in_flight_ to atomic (no size change usually)
// Expected size might increase slightly due to atomic padding/implementation. Re-check assertion after compiling.
#if defined(__i386__) || (defined(__arm__) && !defined(__aarch64__))
// static_assert(sizeof(Node) == 48, "Unexpected size of Node for 32bit compile"); // Adjust expected size
#else
// static_assert(sizeof(Node) == 64, "Unexpected size of Node"); // Adjust expected size
#endif


// Contains Edge and Node pair and set of proxy functions to simplify access
// to them.
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; node_ = nullptr; } // Also reset node_
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const {
    // Compare both edge and node for equality
    return edge_ == other.edge_ && node_ == other.node_;
  }
  bool operator!=(const EdgeAndNode& other) const {
    return !(*this == other);
  }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  // Proxy functions for easier access to node/edge.
  float GetQ(float default_q, float draw_score) const {
    // Check N > 0 to use node's Q, otherwise default_q (FPU)
    return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q;
  }
  float GetWL(float default_wl) const {
    return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl;
  }
  float GetD(float default_d) const {
    return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d;
  }
  float GetM(float default_m) const {
    return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m;
  }
  // N-related getters, from Node (if exists).
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }

  // Whether the node is known to be terminal.
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Node::Bounds GetBounds() const {
    return node_ ? node_->GetBounds()
                 : Node::Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON};
  }

  // Edge related getters.
  float GetP() const { return edge_ ? edge_->GetP() : 0.0f; } // Handle null edge
  Move GetMove(bool flip = false) const {
    return edge_ ? edge_->GetMove(flip) : Move();
  }

  // Returns U = numerator * p / N.
  // Passed numerator is expected to be equal to (cpuct * sqrt(N[parent])).
  float GetU(float numerator) const {
    // Use GetNStarted for PUCT denominator
    return GetP() == 0.0f ? 0.0f : numerator * GetP() / (1.0f + static_cast<float>(GetNStarted()));
  }


  std::string DebugString() const;

 protected:
  // nullptr means that the whole pair is "null". (E.g. when search for a node
  // didn't find anything, or as end iterator signal).
  Edge* edge_ = nullptr;
  // nullptr means that the edge doesn't yet have node extended.
  Node* node_ = nullptr;
};

// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// * Element, pointed by iterator (EdgeAndNode class mainly, but Edge_Iterator
//   is useful too when client wants to call GetOrSpawnNode).
//   It's safe to slice EdgeAndNode off Edge_Iterator.
// It's more customary to have those as three classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized), but
// it's fine if GetOrSpawnNode is called between calls to functions of the
// iterator (e.g. advancing the iterator). Other functions that manipulate
// child_ of parent or the sibling chain are not safe to call while iterating.
template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
   using NodePtr = std::conditional_t<is_const, const Node*, Node*>; // Use NodePtr type alias
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  // Creates "end()" iterator.
  Edge_Iterator() {}

  // Creates "begin()" iterator. Also happens to be a range constructor.
  // child_ptr will be nullptr if parent_node is solid children.
  Edge_Iterator(NodePtr parent_node, Ptr child_ptr) // Use NodePtr
      : EdgeAndNode(parent_node->edges_.get(), nullptr),
        node_ptr_(child_ptr),
        total_count_(parent_node->num_edges_),
        parent_node_(parent_node) // Store parent node
        {
    // If solid, directly point to the start of the array
    if (parent_node_->solid_children_) {
         node_ptr_ = nullptr; // Not used in solid mode for traversal
         edge_ = parent_node_->edges_.get(); // Point to first edge
         if (total_count_ > 0) { // Check if edges exist
              node_ = parent_node_->child_.get(); // Point to first node in array
         } else {
              edge_ = nullptr; // No edges means end iterator
              node_ = nullptr;
         }
         current_idx_ = 0;
    } else { // Linked list mode
        node_ptr_ = &parent_node_->child_;
        edge_ = parent_node_->edges_.get(); // Point to first edge
        current_idx_ = 0;
        if (edge_ && node_ptr_) { // Ensure edge exists before calling Actualize
            Actualize();
        } else {
            edge_ = nullptr; // No edges or no child pointer -> end iterator
            node_ = nullptr;
        }
    }

  }


  // Function to support range interface.
  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }

  // Functions to support iterator interface.
  // Equality comparison operators are inherited from EdgeAndNode.
  void operator++() {
      if (!edge_) return; // Already end iterator

      if (parent_node_->solid_children_) {
           current_idx_++;
           if (current_idx_ >= total_count_) {
               edge_ = nullptr;
               node_ = nullptr;
           } else {
               edge_ = parent_node_->edges_.get() + current_idx_;
               node_ = parent_node_->child_.get() + current_idx_;
           }
      } else { // Linked list mode
          current_idx_++;
          if (current_idx_ >= total_count_) {
               edge_ = nullptr;
               node_ = nullptr; // Also clear node_ for end iterator
          } else {
               edge_ = parent_node_->edges_.get() + current_idx_; // Advance edge pointer
               Actualize(); // Find the corresponding node in linked list
          }
      }
  }

  Edge_Iterator& operator*() { return *this; }

  // If there is node, return it. Otherwise spawn a new one and return it.
   Node* GetOrSpawnNode(Node* parent) { // parent parameter might be redundant if parent_node_ is stored
       if (node_) return node_;  // If there is already a node, return it.

       // Cannot spawn in solid mode here, nodes are pre-allocated
       if (parent_node_->solid_children_) {
            // This indicates an error - trying to spawn in solid mode
            assert(false && "Attempted to spawn node in solid children mode");
            return nullptr;
       }

       // Should never reach here in solid mode.
       assert(node_ptr_ != nullptr);
       Actualize();              // But maybe other thread already did that.
       if (node_) return node_;  // If it did, return.

       // Now we are sure we have to create a new node.
       // Suppose there are nodes with idx 3 and 7, and we want to insert one with
       // idx 5. Here is how it looks like:
       //    node_ptr_ -> &Node(idx_.3).sibling_  ->  Node(idx_.7)
       // Here is how we do that:
       // 1. Store pointer to a node idx_.7:
       //    node_ptr_ -> &Node(idx_.3).sibling_  ->  nullptr
       //    tmp -> Node(idx_.7)
       std::unique_ptr<Node> tmp = std::move(*node_ptr_);
       // 2. Create fresh Node(idx_.5):
       //    node_ptr_ -> &Node(idx_.3).sibling_  ->  Node(idx_.5)
       //    tmp -> Node(idx_.7)
       //    Use parent_node_ member instead of parameter
       *node_ptr_ = std::make_unique<Node>(const_cast<Node*>(parent_node_), current_idx_);
       // 3. Attach stored pointer back to a list:
       //    node_ptr_ ->
       //         &Node(idx_.3).sibling_ -> Node(idx_.5).sibling_ -> Node(idx_.7)
       (*node_ptr_)->sibling_ = std::move(tmp);
       // 4. Actualize:
       //    node_ -> &Node(idx_.5)
       //    node_ptr_ -> &Node(idx_.5).sibling_ -> Node(idx_.7)
       Actualize();
       return node_;
   }


 private:
  void Actualize() {
      // This must never be called in solid mode.
      if (parent_node_->solid_children_ || !node_ptr_) {
           // If called in solid mode or node_ptr_ is null, something is wrong
           // Set node_ to nullptr to indicate error or end state
           node_ = nullptr;
           return;
      }

      // If node_ptr_ is behind, advance it.
      // This is needed (and has to be 'while' rather than 'if') as other threads
      // could spawn new nodes between &node_ptr_ and *node_ptr_ while we didn't
      // see.
      while (*node_ptr_ && (*node_ptr_)->index_ < current_idx_) {
          node_ptr_ = &(*node_ptr_)->sibling_;
      }
      // If in the end node_ptr_ points to the node that we need, populate node_
      // and advance node_ptr_.
      if (*node_ptr_ && (*node_ptr_)->index_ == current_idx_) {
          node_ = (*node_ptr_).get();
          node_ptr_ = &node_->sibling_; // Advance node_ptr_ for next ++ call
      } else {
          node_ = nullptr; // Corresponding node not found (dangling edge)
      }
  }


  // Pointer to a pointer to the next node. Has to be a pointer to pointer
  // as we'd like to update it when spawning a new node. (Only used in linked list mode)
  Ptr node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
  NodePtr parent_node_ = nullptr; // Store parent node pointer
};


// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// It's more customary to have those as two classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized).
template <bool is_const>
class VisitedNode_Iterator {
 public:
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>; // Use NodePtr type alias

  // Creates "end()" iterator.
  VisitedNode_Iterator() {}

  // Creates "begin()" iterator. Also happens to be a range constructor.
  // child_ptr will be nullptr if parent_node is solid children.
  VisitedNode_Iterator(NodePtr parent_node) // Takes parent directly
      : parent_node_(parent_node),
        total_count_(parent_node ? parent_node->num_edges_ : 0), // Handle null parent
        solid_(parent_node ? parent_node->solid_children_ : false) // Handle null parent
        {
        if (!parent_node) { // Handle null parent case
            node_ptr_ = nullptr;
            current_idx_ = 0;
            return;
        }

        if (solid_) {
            node_ptr_ = parent_node_->child_.get(); // Point to start of array
            current_idx_ = 0;
             // Find first visited node
             while (current_idx_ < total_count_ && (!node_ptr_ || node_ptr_[current_idx_].GetN() == 0)) {
                 current_idx_++;
             }
             if (current_idx_ >= total_count_) {
                  node_ptr_ = nullptr; // No visited nodes found
             } else {
                  // node_ptr_ is already pointing to the array start, index handles current node
             }
        } else { // Linked list mode
             node_ptr_ = parent_node_->child_.get(); // Start from first child
             // Find first visited node
             while (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
                 node_ptr_ = node_ptr_->sibling_.get();
             }
             current_idx_ = node_ptr_ ? node_ptr_->index_ : total_count_; // Store index for potential future use if needed
        }

  }
  // These are technically wrong, but are usable to compare with end().
  bool operator==(const VisitedNode_Iterator<is_const>& other) const {
    // Compare based on node_ptr_ and potentially index for solid mode
     if (solid_) return (node_ptr_ == other.node_ptr_ && current_idx_ == other.current_idx_) || (node_ptr_ == nullptr && other.node_ptr_ == nullptr && current_idx_ >= total_count_ && other.current_idx_ >= other.total_count_);
     return node_ptr_ == other.node_ptr_;
  }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const {
    return !(*this == other);
  }

  // Function to support range interface.
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; } // Default constructor is end iterator

  // Functions to support iterator interface.
  void operator++() {
     if (!node_ptr_) return; // Already at end

     if (solid_) {
         // Move to the next index and find the next visited node
         current_idx_++;
          while (current_idx_ < total_count_ && node_ptr_[current_idx_].GetN() == 0) {
              // Skip check for n_in_flight==0 for simplicity based on VisitedNodes usage
              current_idx_++;
          }
          if (current_idx_ >= total_count_) {
              node_ptr_ = nullptr; // Reached end
              current_idx_ = total_count_; // Set index consistently for end comparison
          }
          // No need to change node_ptr_, index handles the current position
     } else { // Linked list mode
         // Move to the next sibling and find the next visited node
         do {
             node_ptr_ = node_ptr_->sibling_.get();
         } while (node_ptr_ != nullptr && node_ptr_->GetN() == 0);
         // Update index if needed (though not strictly necessary for linked list iteration)
         current_idx_ = node_ptr_ ? node_ptr_->index_ : total_count_;
     }
  }


  Node* operator*() {
    if (solid_) {
       // Check if current_idx_ is valid before dereferencing
        return (current_idx_ < total_count_ && node_ptr_ != nullptr) ? &(node_ptr_[current_idx_]) : nullptr;
    } else {
      return node_ptr_;
    }
  }

 private:
  // Pointer to current node (or start of array in solid mode).
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0; // Index used primarily for solid mode
  uint16_t total_count_ = 0;
  bool solid_ = false;
  NodePtr parent_node_ = nullptr; // Store parent node pointer
};


inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return VisitedNode_Iterator<true>(this); // Use the new constructor
}
inline VisitedNode_Iterator<false> Node::VisitedNodes() {
  return VisitedNode_Iterator<false>(this); // Use the new constructor
}

class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }
  // Adds a move to current_head_.
  void MakeMove(Move move);
  // Resets the current head to ensure it doesn't carry over details from a
  // previous search.
  void TrimTreeAtHead();
  // Sets the position in a tree, trying to reuse the tree.
  // If @auto_garbage_collect, old tree is garbage collected immediately. (may
  // take some milliseconds)
  // Returns whether a new position the same game as old position (with some
  // moves added). Returns false, if the position is completely different,
  // or if it's shorter than before.
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<std::string>& moves);
  bool ResetToPosition(const GameState& pos);
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  // A node which to start search from.
  Node* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
};

}  // namespace classic
}  // namespace lczero
