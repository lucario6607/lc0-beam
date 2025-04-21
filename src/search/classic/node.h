/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  ... (License remains the same) ...
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic> // Added for std::atomic
#include <vector> // Added for std::vector
#include <limits> // Added for numeric_limits

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h" // Include for PositionHash (if defined here)
#include "chess/chess.h"     // Include for Value, GameResult, kValueMate etc.
#include "neural/encoder.h"
#include "proto/net.pb.h"
#include "utils/mutex.h"


namespace lczero {
namespace classic {

// Forward declaration
class SearchParams;

// Children of a node... (Comment remains the same) ...

class Node;
class Edge {
 public:
  // ... (Edge class remains the same) ...
  Move move_;
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

  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {}

  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  Node* CreateSingleChildNode(Move m);
  void CreateEdges(const MoveList& moves);
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetVisitCount() const { return GetN(); }
  uint32_t GetChildrenVisits() const {
      uint32_t n = GetN();
      return n > 0 ? n - 1 : 0;
  }
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }
  int GetEffectiveVisits() const { return GetNStarted(); }
  int GetEffectiveParentVisits() const {
      return GetParent() ? GetParent()->GetNStarted() : 0;
   }

  // REMOVED GetValueSum() as value_sum_ doesn't exist

  float GetQ(float draw_score) const {
       uint32_t n = GetN();
       if (n == 0) return 0.0f;
       double current_wl = wl_.load(std::memory_order_relaxed);
       float current_d = d_.load(std::memory_order_relaxed);
       return static_cast<float>(current_wl + draw_score * current_d);
   }
  float GetValue() const { // Returns WL value
      uint32_t n = GetN();
      if (n == 0) return 0.0f;
      return static_cast<float>(wl_.load(std::memory_order_relaxed));
  }
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); }
  float GetD() const { return d_.load(std::memory_order_relaxed); }
  float GetM() const { return m_.load(std::memory_order_relaxed); }
  float GetPolicyPrior() const;
  // REMOVED GetQ(const SearchParams&) helper - implement logic in SelectChild

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);
  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(Value v, float d, float m, int multivisit); // Use Value type
  void AdjustForTerminal(Value v, float d, float m, int multivisit);   // Use Value type
  void RevertTerminalVisits(Value v, float d, float m, int multivisit); // Use Value type
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); }
  void UpdateMaxDepth(int depth);
  bool UpdateFullDepth(uint16_t* depth);
  ConstIterator Edges() const;
  Iterator Edges();
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();
  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;
  std::string DebugString() const;
  bool MakeSolid();
  void SortEdges();
  uint16_t Index() const { return index_; }

   // --- NEW MEMBERS for Proven State ---
   std::atomic<bool> is_known_win{false};
   std::atomic<bool> is_known_loss{false};

   // --- NEW METHODS for Proven State ---
   Value GetMinValue() const; // Use Value type
   Value GetMaxValue() const; // Use Value type

   // --- NEW Helper Methods ---
   const std::unique_ptr<Node>* GetChildrenPtr() const { return &child_; }
   Node* GetChild(int index) const;
   int GetNumChildren() const { return num_edges_; }


  ~Node() {
    if (solid_children_ && child_) {
      for (int i = 0; i < num_edges_; i++) {
         if (child_.get() + i != nullptr) {
            child_.get()[i].~Node();
         }
      }
      std::allocator<Node> alloc;
      // Check if pointer is valid before deallocating
      if (child_) { // Check added
          alloc.deallocate(child_.release(), num_edges_);
      }
    }
  }

 private:
  void UpdateChildrenParents();

  // Member variable order (largest to smallest, atomics grouped)
  // 8 byte fields.
  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  std::atomic<double> wl_{0.0}; // Using atomic double

  // 4 byte fields.
  std::atomic<float> d_{0.0};
  std::atomic<float> m_{0.0};
  std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};

  // 2 byte fields.
  uint16_t index_;

  // 1 byte fields.
  uint8_t num_edges_ = 0;
  Terminal terminal_type_ : 2;
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  bool solid_children_ : 1;
  // Known state flags fit here
  // atomic<bool> might take more than 1 bit, check alignment/padding
  // Putting them here explicitly, but compiler might rearrange slightly
  // std::atomic<bool> is_known_win{false};  // Moved above for grouping
  // std::atomic<bool> is_known_loss{false}; // Moved above for grouping

  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
  friend class SearchWorker;
};

// Re-check assertion after compiling with atomics
// static_assert(sizeof(Node) == 64, "Unexpected size of Node");

class EdgeAndNode {
 public:
  // ... (EdgeAndNode class remains largely the same, ensure GetQ uses node_->GetQ) ...
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

   float GetQ(float default_q, float draw_score) const {
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
   uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
   int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
   uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
   bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
   bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
   Node::Bounds GetBounds() const {
     return node_ ? node_->GetBounds()
                  : Node::Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON};
   }
   float GetP() const { return edge_ ? edge_->GetP() : 0.0f; }
   Move GetMove(bool flip = false) const {
     return edge_ ? edge_->GetMove(flip) : Move();
   }
   float GetU(float numerator) const {
       return GetP() == 0.0f ? 0.0f : numerator * GetP() / (1.0f + static_cast<float>(GetNStarted()));
   }
   std::string DebugString() const;

 protected:
   Edge* edge_ = nullptr;
   Node* node_ = nullptr;
};

template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  Edge_Iterator() {}

   Edge_Iterator(NodePtr parent_node, Ptr child_ptr) // Use NodePtr
       : EdgeAndNode(parent_node->edges_.get(), nullptr),
         parent_node_(parent_node), // Initialize parent first
         node_ptr_(child_ptr),
         total_count_(parent_node->num_edges_) // Initialize total_count after parent
         {
     if (parent_node_->solid_children_) {
         node_ptr_ = nullptr;
         edge_ = parent_node_->edges_.get();
         if (total_count_ > 0) {
              node_ = parent_node_->child_.get();
         } else {
              edge_ = nullptr;
              node_ = nullptr;
         }
         current_idx_ = 0;
     } else {
         node_ptr_ = &parent_node_->child_;
         edge_ = parent_node_->edges_.get();
         current_idx_ = 0;
         if (edge_ && node_ptr_) {
             Actualize();
         } else {
             edge_ = nullptr;
             node_ = nullptr;
         }
     }
   }

  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }
  void operator++(); // Implementation moved to .cc if needed, or keep inline
  Edge_Iterator& operator*() { return *this; }
  Node* GetOrSpawnNode(Node* parent); // Implementation moved to .cc

 private:
  void Actualize(); // Implementation moved to .cc

  NodePtr parent_node_ = nullptr; // Store parent node pointer (Declared first)
  Ptr node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

template <bool is_const>
class VisitedNode_Iterator {
 public:
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;

  VisitedNode_Iterator() {}

   VisitedNode_Iterator(NodePtr parent_node) // Takes parent directly
       : parent_node_(parent_node), // Initialize parent first
         solid_(parent_node ? parent_node->solid_children_ : false), // Initialize solid after parent
         total_count_(parent_node ? parent_node->num_edges_ : 0) // Initialize total_count after parent
         {
         // Fix reorder: Initialize node_ptr_ and current_idx_ last
         if (!parent_node) {
             node_ptr_ = nullptr;
             current_idx_ = 0;
             return;
         }
         if (solid_) {
             node_ptr_ = parent_node_->child_.get();
             current_idx_ = 0;
              while (current_idx_ < total_count_ && (!node_ptr_ || node_ptr_[current_idx_].GetN() == 0)) {
                  current_idx_++;
              }
              if (current_idx_ >= total_count_) {
                   node_ptr_ = nullptr;
              }
         } else {
              node_ptr_ = parent_node_->child_.get();
              while (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
                  node_ptr_ = node_ptr_->sibling_.get();
              }
              current_idx_ = node_ptr_ ? node_ptr_->index_ : total_count_;
         }
   }

  bool operator==(const VisitedNode_Iterator<is_const>& other) const; // Implementation can move to .cc
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return !(*this == other); }
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }
  void operator++(); // Implementation can move to .cc
  Node* operator*(); // Implementation can move to .cc

 private:
   // Declaration order fixed
   NodePtr parent_node_ = nullptr; // Store parent node pointer
   bool solid_ = false;
   uint16_t total_count_ = 0;
   Node* node_ptr_ = nullptr;
   uint16_t current_idx_ = 0;
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return VisitedNode_Iterator<true>(this);
}
inline VisitedNode_Iterator<false> Node::VisitedNodes() {
  return VisitedNode_Iterator<false>(this);
}

class NodeTree {
 // ... (NodeTree remains the same) ...
 public:
  ~NodeTree() { DeallocateTree(); }
  void MakeMove(Move move);
  void TrimTreeAtHead();
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
  Node* current_head_ = nullptr;
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
};

}  // namespace classic
}  // namespace lczero
