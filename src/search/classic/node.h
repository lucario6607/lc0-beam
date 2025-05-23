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

// --- Core Type Definitions FIRST ---
#include "chess/types.h"     // Defines lczero::Value, lczero::GameResult, lczero::Move, lczero::MoveList etc.

// --- Standard Library Includes ---
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex> // Required by utils/mutex.h potentially
#include <string>
#include <utility>
#include <vector>
#include <type_traits> // For std::conditional_t

// --- Other Necessary LC0 Includes ---
#include "utils/mutex.h"    // Defines Mutex, SharedMutex etc.

// --- Forward Declarations ---
namespace lczero {
class Position; // Forward declare if only used via pointer/ref in header
class PositionHistory;

namespace classic {
class NodeTree;
class SearchParams;
class Node;
class Edge;
class EdgeAndNode;
template <bool is_const> class Edge_Iterator;
template <bool is_const> class VisitedNode_Iterator;
class SearchWorker;
} // namespace classic
} // namespace lczero


namespace lczero {
namespace classic {

// --- Edge ---
class Edge {
 public:
  static std::unique_ptr<Edge[]> FromMovelist(const lczero::MoveList& moves);
  lczero::Move GetMove(bool as_opponent = false) const;
  float GetP() const;
  void SetP(float val);
  std::string DebugString() const;

 private:
  lczero::Move move_;
  uint16_t p_ = 0;
  friend class Node;
  template <bool is_const> friend class Edge_Iterator;
};

// --- Node ---
class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;
  using VisitedIterator = VisitedNode_Iterator<false>;
  using ConstVisitedIterator = VisitedNode_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  Node(Node* parent, uint16_t index);
  ~Node();
  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  Node* CreateSingleChildNode(lczero::Move m);
  void CreateEdges(const lczero::MoveList& moves);
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;

  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetVisitCount() const { return GetN(); }
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetChildrenVisits() const { uint32_t n = GetN(); return n > 0 ? n - 1 : 0; }
  int GetEffectiveVisits() const { return GetNStarted(); }
  int GetEffectiveParentVisits() const;

  float GetQ(float draw_score) const;
  lczero::Value GetValue() const; // <<< Uses lczero::Value
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); }
  float GetD() const { return d_.load(std::memory_order_relaxed); }
  float GetM() const { return m_.load(std::memory_order_relaxed); }
  float GetPolicyPrior() const;

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<lczero::GameResult, lczero::GameResult> Bounds; // <<< Uses lczero::GameResult
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  void SetBounds(lczero::GameResult lower, lczero::GameResult upper); // <<< Uses lczero::GameResult
  void MakeTerminal(lczero::GameResult result, float plies_left = 0.0f, // <<< Uses lczero::GameResult
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();

  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;

  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(lczero::Value v, float d, float m, int multivisit); // <<< Uses lczero::Value
  void AdjustForTerminal(lczero::Value v_delta, float d_delta, float m_delta, int multivisit); // <<< Uses lczero::Value
  void RevertTerminalVisits(lczero::Value v, float d, float m, int multivisit); // <<< Uses lczero::Value
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); }

  ConstIterator Edges() const;
  Iterator Edges();
  ConstVisitedIterator VisitedNodes() const;
  VisitedIterator VisitedNodes();

  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node_to_save);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;
  std::string DebugString() const;
  bool MakeSolid();
  void SortEdges();

  uint16_t Index() const { return index_; }

   std::atomic<bool> is_known_win{false};
   std::atomic<bool> is_known_loss{false};
   lczero::Value GetMinValue() const; // <<< Uses lczero::Value
   lczero::Value GetMaxValue() const; // <<< Uses lczero::Value

   Node* GetChild(int index) const;

 private:
  void UpdateChildrenParents();

  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  std::atomic<double> wl_{0.0}; // Use double for higher precision accumulation

  std::atomic<float> d_{0.0};
  std::atomic<float> m_{0.0};
  std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};

  uint16_t index_;

  uint8_t num_edges_ = 0;
  Terminal terminal_type_ : 2;
  lczero::GameResult lower_bound_ : 2; // <<< Uses lczero::GameResult
  lczero::GameResult upper_bound_ : 2; // <<< Uses lczero::GameResult
  bool solid_children_ : 1;

  friend class NodeTree;
  template <bool is_const> friend class Edge_Iterator;
  template <bool is_const> friend class VisitedNode_Iterator;
  friend class SearchWorker;
};

// --- EdgeAndNode ---
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; node_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
   bool operator==(const EdgeAndNode& other) const { return edge_ == other.edge_ && node_ == other.node_; }
   bool operator!=(const EdgeAndNode& other) const { return !(*this == other); }
   bool HasNode() const { return node_ != nullptr; }
   Edge* edge() const { return edge_; }
   Node* node() const { return node_; }

   float GetQ(float default_q, float draw_score) const;
   float GetWL(float default_wl) const;
   float GetD(float default_d) const;
   float GetM(float default_m) const;
   uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
   int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
   uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
   bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
   bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
   Node::Bounds GetBounds() const; // <<< Uses Node::Bounds -> lczero::GameResult
   float GetP() const { return edge_ ? edge_->GetP() : 0.0f; }
   lczero::Move GetMove(bool flip = false) const; // <<< Uses lczero::Move
   float GetU(float numerator) const;
   std::string DebugString() const;

 protected:
   Edge* edge_ = nullptr;
   Node* node_ = nullptr;
};

// --- Edge_Iterator ---
template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*, std::unique_ptr<Node>*>;
  using value_type = EdgeAndNode;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = const EdgeAndNode*;
  using reference = const EdgeAndNode&;

  Edge_Iterator();
  Edge_Iterator(NodePtr parent_node);

  bool operator==(const Edge_Iterator<is_const>& other) const;
  bool operator!=(const Edge_Iterator<is_const>& other) const { return !(*this == other); }

  Edge_Iterator<is_const>& begin();
  Edge_Iterator<is_const> end();
  Edge_Iterator<is_const>& operator++();
  reference operator*() const;
  pointer operator->() const;

  Node* GetOrSpawnNode(Node* parent);

 private:
  void Actualize();

  NodePtr parent_node_ = nullptr;
  Ptr node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
};

// --- VisitedNode_Iterator ---
template <bool is_const>
class VisitedNode_Iterator {
 public:
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using value_type = NodePtr;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = const NodePtr*;
  using reference = const NodePtr&;

  VisitedNode_Iterator();
  VisitedNode_Iterator(NodePtr parent_node);

  bool operator==(const VisitedNode_Iterator<is_const>& other) const;
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return !(*this == other); }

  VisitedNode_Iterator<is_const>& begin();
  VisitedNode_Iterator<is_const> end();
  VisitedNode_Iterator<is_const>& operator++();
  NodePtr operator*() const;

 private:
   void AdvanceToNextVisited();

   NodePtr parent_node_ = nullptr;
   NodePtr current_node_ptr_ = nullptr;
   uint16_t current_idx_ = 0;
   uint16_t total_count_ = 0;
   bool solid_ = false;
};


// --- Inline Implementations ---
inline Node::ConstIterator Node::Edges() const { return ConstIterator(this); }
inline Node::Iterator Node::Edges() { return Iterator(this); }
inline Node::ConstVisitedIterator Node::VisitedNodes() const { return ConstVisitedIterator(this); }
inline Node::VisitedIterator Node::VisitedNodes() { return VisitedIterator(this); }

template <bool is_const> inline Edge_Iterator<is_const>& Edge_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline Edge_Iterator<is_const> Edge_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline typename Edge_Iterator<is_const>::reference Edge_Iterator<is_const>::operator*() const { return *this; }
template <bool is_const> inline typename Edge_Iterator<is_const>::pointer Edge_Iterator<is_const>::operator->() const { return this; }
template <bool is_const> inline bool Edge_Iterator<is_const>::operator==(const Edge_Iterator<is_const>& other) const { return edge_ == other.edge_; }

template <bool is_const> inline VisitedNode_Iterator<is_const>& VisitedNode_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline VisitedNode_Iterator<is_const> VisitedNode_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline typename VisitedNode_Iterator<is_const>::NodePtr VisitedNode_Iterator<is_const>::operator*() const { return current_node_ptr_; }
template <bool is_const> inline bool VisitedNode_Iterator<is_const>::operator==(const VisitedNode_Iterator<is_const>& other) const { return current_node_ptr_ == other.current_node_ptr_; }

inline Node::Node(Node* parent, uint16_t index)
    : parent_(parent),
      index_(index),
      terminal_type_(Terminal::NonTerminal),
      lower_bound_(lczero::GameResult::BLACK_WON),
      upper_bound_(lczero::GameResult::WHITE_WON),
      solid_children_(false) {}

#if defined(_M_IX86) || defined(__i386__) || (defined(__arm__) && !defined(__aarch64__))
static_assert(sizeof(Node) == 48 || sizeof(Node) == 52, "Unexpected size of Node for 32bit compile");
#else
static_assert(sizeof(Node) == 72, "Unexpected size of Node");
#endif

}  // namespace classic
}  // namespace lczero
