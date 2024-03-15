/*!
 * Copyright 2014-2019 by Contributors
 * Modifications Copyright 2020-22 by Secure XGBoost Contributors
 * \file tree_model.h
 * \brief model structure for tree
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_MODEL_H_
#define XGBOOST_TREE_MODEL_H_

#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/feature_map.h>
#include <xgboost/logging.h>
#include <xgboost/model.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

#ifdef __ENCLAVE_OBLIVIOUS__
#include "enclave/dpobl_operator.h"
#include "enclave/obl_primitives.h"
#endif

namespace xgboost {

struct PathElement;  // forward declaration

class Json;
// FIXME(trivialfis): Once binary IO is gone, make this parameter internal as it
// should not be configured by users.
/*! \brief meta parameters of the tree */
struct TreeParam : public dmlc::Parameter<TreeParam> {
  /*! \brief (Deprecated) number of start root */
  int deprecated_num_roots;
  /*! \brief total number of nodes */
  int num_nodes;
  /*!\brief number of deleted nodes */
  int num_deleted;
  /*! \brief maximum depth, this is a statistics of the tree */
  int deprecated_max_depth;
  /*! \brief number of features used for tree construction */
  int num_feature;
  /*!
   * \brief leaf vector size, used for vector tree
   * used to store more than one dimensional information in tree
   */
  int size_leaf_vector;
  /*! \brief reserved part, make sure alignment works for 64bit */
  int reserved[31];
  /*! \brief constructor */
  TreeParam() {
    // assert compact alignment
    static_assert(sizeof(TreeParam) == (31 + 6) * sizeof(int),
                  "TreeParam: 64 bit align");
    std::memset(this, 0, sizeof(TreeParam));
    num_nodes = 1;
    deprecated_num_roots = 1;
  }
  // declare the parameters
  DMLC_DECLARE_PARAMETER(TreeParam) {
    // only declare the parameters that can be set by the user.
    // other arguments are set by the algorithm.
    DMLC_DECLARE_FIELD(num_nodes).set_lower_bound(1).set_default(1);
    DMLC_DECLARE_FIELD(num_feature)
        .describe("Number of features used in tree construction.");
    DMLC_DECLARE_FIELD(num_deleted);
    DMLC_DECLARE_FIELD(size_leaf_vector)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Size of leaf vector, reserved for vector tree");
  }

  bool operator==(const TreeParam& b) const {
    return num_nodes == b.num_nodes && num_deleted == b.num_deleted &&
           num_feature == b.num_feature &&
           size_leaf_vector == b.size_leaf_vector;
  }
};

/*! \brief node statistics used in regression tree */
struct RTreeNodeStat {
  /*! \brief loss change caused by current split */
  bst_float loss_chg;
  /*! \brief sum of hessian values, used to measure coverage of data */
  bst_float sum_hess;
  /*! \brief weight of current node */
  bst_float base_weight;
  /*! \brief number of child that is leaf node known up to now */
  int leaf_child_cnt{0};

  RTreeNodeStat() = default;
  RTreeNodeStat(float loss_chg, float sum_hess, float weight)
      : loss_chg{loss_chg}, sum_hess{sum_hess}, base_weight{weight} {}
  bool operator==(const RTreeNodeStat& b) const {
    return loss_chg == b.loss_chg && sum_hess == b.sum_hess &&
           base_weight == b.base_weight && leaf_child_cnt == b.leaf_child_cnt;
  }
};

/*!
 * \brief define regression tree to be the most common tree model.
 *  This is the data structure used in xgboost's major tree models.
 */
class RegTree : public Model {
 public:
  using SplitCondT = bst_float;
  static constexpr bst_node_t kInvalidNodeId{-1};
  static constexpr uint32_t kDeletedNodeMarker =
      std::numeric_limits<uint32_t>::max();
  static constexpr bst_node_t kRoot{0};

  /*! \brief tree node */
  class Node {
   public:
    XGBOOST_DEVICE Node() {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    Node(int32_t cleft, int32_t cright, int32_t parent, uint32_t split_ind,
         float split_cond, bool default_left)
        : parent_{parent}, cleft_{cleft}, cright_{cright} {
      this->SetParent(parent_);
      this->SetSplit(split_ind, split_cond, default_left);
    }

    /*! \brief index of left child */
    XGBOOST_DEVICE int LeftChild() const { return this->cleft_; }
    /*! \brief index of right child */
    XGBOOST_DEVICE int RightChild() const { return this->cright_; }
    /*! \brief index of default child when feature is missing */
    XGBOOST_DEVICE int DefaultChild() const {
      return this->DefaultLeft() ? this->LeftChild() : this->RightChild();
    }
    /*! \brief feature index of split condition */
    XGBOOST_DEVICE unsigned SplitIndex() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    XGBOOST_DEVICE bool DefaultLeft() const { return (sindex_ >> 31) != 0; }
    /*! \brief whether current node is leaf node */
    XGBOOST_DEVICE bool IsLeaf() const { return cleft_ == kInvalidNodeId; }
    /*! \return get leaf value of leaf node */
    XGBOOST_DEVICE bst_float LeafValue() const {
      return (this->info_).leaf_value;
    }
    /*! \return get split condition of the node */
    XGBOOST_DEVICE SplitCondT SplitCond() const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    XGBOOST_DEVICE int Parent() const { return parent_ & ((1U << 31) - 1); }
    /*! \brief whether current node is left child */
    XGBOOST_DEVICE bool IsLeftChild() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether this node is deleted */
    XGBOOST_DEVICE bool IsDeleted() const {
      return sindex_ == kDeletedNodeMarker;
    }
    /*! \brief whether current node is root */
    XGBOOST_DEVICE bool IsRoot() const { return parent_ == kInvalidNodeId; }
    /*!
     * \brief set the left child
     * \param nid node id to right child
     */
    XGBOOST_DEVICE void SetLeftChild(int nid) { this->cleft_ = nid; }
    /*!
     * \brief set the right child
     * \param nid node id to right child
     */
    XGBOOST_DEVICE void SetRightChild(int nid) { this->cright_ = nid; }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    XGBOOST_DEVICE void SetSplit(unsigned split_index, SplitCondT split_cond,
                                 bool default_left = false) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).split_cond = split_cond;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     * \param right right index, could be used to store
     *        additional information
     */
    XGBOOST_DEVICE void SetLeaf(bst_float value, int right = kInvalidNodeId) {
      (this->info_).leaf_value = value;
      this->cleft_ = kInvalidNodeId;
      this->cright_ = right;
    }
    /*! \brief mark that this node is deleted */
    XGBOOST_DEVICE void MarkDelete() { this->sindex_ = kDeletedNodeMarker; }
    /*! \brief Reuse this deleted node. */
    XGBOOST_DEVICE void Reuse() { this->sindex_ = 0; }
    // set parent
    XGBOOST_DEVICE void SetParent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
    bool operator==(const Node& b) const {
      return parent_ == b.parent_ && cleft_ == b.cleft_ &&
             cright_ == b.cright_ && sindex_ == b.sindex_ &&
             info_.leaf_value == b.info_.leaf_value;
    }

   private:
    /*!
     * \brief in leaf node, we have weights, in non-leaf nodes,
     *        we have split condition
     */
    union Info {
      bst_float leaf_value;
      SplitCondT split_cond;
    };
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int32_t parent_{kInvalidNodeId};
    // pointer to left, right
    int32_t cleft_{kInvalidNodeId}, cright_{kInvalidNodeId};
    // split feature index, left split or right split depends on the highest bit
    uint32_t sindex_{0};
    // extra info
    Info info_;
  };

  /*!
   * \brief change a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  void ChangeToLeaf(int rid, bst_float value) {
    CHECK(nodes_[nodes_[rid].LeftChild()].IsLeaf());
    CHECK(nodes_[nodes_[rid].RightChild()].IsLeaf());
    this->DeleteNode(nodes_[rid].LeftChild());
    this->DeleteNode(nodes_[rid].RightChild());
    nodes_[rid].SetLeaf(value);
  }
  /*!
   * \brief collapse a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  void CollapseToLeaf(int rid, bst_float value) {
    if (nodes_[rid].IsLeaf()) return;
    if (!nodes_[nodes_[rid].LeftChild()].IsLeaf()) {
      CollapseToLeaf(nodes_[rid].LeftChild(), 0.0f);
    }
    if (!nodes_[nodes_[rid].RightChild()].IsLeaf()) {
      CollapseToLeaf(nodes_[rid].RightChild(), 0.0f);
    }
    this->ChangeToLeaf(rid, value);
  }

  /*! \brief model parameter */
  TreeParam param;
  /*! \brief constructor */
  RegTree() {
    param.num_nodes = 1;
    param.num_deleted = 0;
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    for (int i = 0; i < param.num_nodes; i++) {
      nodes_[i].SetLeaf(0.0f);
      nodes_[i].SetParent(kInvalidNodeId);
    }
#ifdef __ENCLAVE_DPOBLIVIOUS__
    stash_.SetTree(this);
#endif
  }
  /*! \brief get node given nid */
  Node& operator[](int nid) { return nodes_[nid]; }
  /*! \brief get node given nid */
  const Node& operator[](int nid) const { return nodes_[nid]; }

  /*! \brief get const reference to nodes */
  const std::vector<Node>& GetNodes() const { return nodes_; }

  /*! \brief get node statistics given nid */
  RTreeNodeStat& Stat(int nid) { return stats_[nid]; }
  /*! \brief get node statistics given nid */
  const RTreeNodeStat& Stat(int nid) const { return stats_[nid]; }

  /*!
   * \brief load model from stream
   * \param fi input stream
   */
  void Load(dmlc::Stream* fi);
  /*!
   * \brief save model to stream
   * \param fo output stream
   */
  void Save(dmlc::Stream* fo) const;

  void LoadModel(Json const& in) override;
  void SaveModel(Json* out) const override;

  bool operator==(const RegTree& b) const {
    return nodes_ == b.nodes_ && stats_ == b.stats_ &&
           deleted_nodes_ == b.deleted_nodes_ && param == b.param;
  }
  /* \brief Iterate through all nodes in this tree.
   *
   * \param Function that accepts a node index, and returns false when iteration
   * should stop, otherwise returns true.
   */
  template <typename Func>
  void WalkTree(Func func) const {
    std::stack<bst_node_t> nodes;
    nodes.push(kRoot);
    auto& self = *this;
    while (!nodes.empty()) {
      auto nidx = nodes.top();
      nodes.pop();
      if (!func(nidx)) {
        return;
      }
      auto left = self[nidx].LeftChild();
      auto right = self[nidx].RightChild();
      if (left != RegTree::kInvalidNodeId) {
        nodes.push(left);
      }
      if (right != RegTree::kInvalidNodeId) {
        nodes.push(right);
      }
    }
  }
  /*!
   * \brief Compares whether 2 trees are equal from a user's perspective.  The
   * equality compares only non-deleted nodes.
   *
   * \parm b The other tree.
   */
  bool Equal(const RegTree& b) const;

  /**
   * \brief Expands a leaf node into two additional leaf nodes.
   *
   * \param nid               The node index to expand.
   * \param split_index       Feature index of the split.
   * \param split_value       The split condition.
   * \param default_left      True to default left.
   * \param base_weight       The base weight, before learning rate.
   * \param left_leaf_weight  The left leaf weight for prediction, modified by
   * learning rate. \param right_leaf_weight The right leaf weight for
   * prediction, modified by learning rate. \param loss_change       The loss
   * change. \param sum_hess          The sum hess. \param left_sum          The
   * sum hess of left leaf. \param right_sum         The sum hess of right leaf.
   * \param leaf_right_child  The right child index of leaf, by default
   * kInvalidNodeId, some updaters use the right child index of leaf as a marker
   */
  void ExpandNode(int nid, unsigned split_index, bst_float split_value,
                  bool default_left, bst_float base_weight,
                  bst_float left_leaf_weight, bst_float right_leaf_weight,
                  bst_float loss_change, float sum_hess, float left_sum,
                  float right_sum,
                  bst_node_t leaf_right_child = kInvalidNodeId) {
    int pleft = this->AllocNode();
    int pright = this->AllocNode();
    auto& node = nodes_[nid];
    CHECK(node.IsLeaf());
    node.SetLeftChild(pleft);
    node.SetRightChild(pright);
    nodes_[node.LeftChild()].SetParent(nid, true);
    nodes_[node.RightChild()].SetParent(nid, false);
    node.SetSplit(split_index, split_value, default_left);

    nodes_[pleft].SetLeaf(left_leaf_weight, leaf_right_child);
    nodes_[pright].SetLeaf(right_leaf_weight, leaf_right_child);

    this->Stat(nid) = {loss_change, sum_hess, base_weight};
    this->Stat(pleft) = {0.0f, left_sum, left_leaf_weight};
    this->Stat(pright) = {0.0f, right_sum, right_leaf_weight};
  }

  /*!
   * \brief get current depth
   * \param nid node id
   */
  int GetDepth(int nid) const {
    int depth = 0;
    while (!nodes_[nid].IsRoot()) {
      ++depth;
      nid = nodes_[nid].Parent();
    }
    return depth;
  }
  /*!
   * \brief get maximum depth
   * \param nid node id
   */
  int MaxDepth(int nid) const {
    if (nodes_[nid].IsLeaf()) return 0;
    return std::max(MaxDepth(nodes_[nid].LeftChild()) + 1,
                    MaxDepth(nodes_[nid].RightChild()) + 1);
  }

  /*!
   * \brief get maximum depth
   */
  int MaxDepth() { return MaxDepth(0); }

  /*! \brief number of extra nodes besides the root */
  int NumExtraNodes() const { return param.num_nodes - 1 - param.num_deleted; }

  /* \brief Count number of leaves in tree. */
  bst_node_t GetNumLeaves() const;
  bst_node_t GetNumSplitNodes() const;

  /*!
   * \brief dense feature vector that can be taken by RegTree
   * and can be construct from sparse feature vector.
   */
  struct FVec {
    /*!
     * \brief initialize the vector with size vector
     * \param size The size of the feature vector.
     */
    void Init(size_t size);
    /*!
     * \brief fill the vector with sparse vector
     * \param inst The sparse instance to fill.
     */
    void Fill(const SparsePage::Inst& inst);
    /*!
     * \brief drop the trace after fill, must be called after fill.
     * \param inst The sparse instance to drop.
     */
    void Drop(const SparsePage::Inst& inst);
    /*!
     * \brief returns the size of the feature vector
     * \return the size of the feature vector
     */
    size_t Size() const;
    void Print() const;
    /*!
     * \brief get ith value
     * \param i feature index.
     * \return the i-th feature value
     */
    bst_float GetFvalue(size_t i) const;
    /*!
     * \brief check whether i-th entry is missing
     * \param i feature index.
     * \return whether i-th value is missing.
     */
    bool IsMissing(size_t i) const;

    /*!
     * \brief a union value of value and flag
     *  when flag == -1, this indicate the value is missing
     */
    union Entry {
      bst_float fvalue;
      int flag;
    };

    std::vector<Entry>& Data();
#ifdef __ENCLAVE_OBLIVIOUS__
    /*!
     * \brief get ith entry obliviously
     * \param i feature index.
     * \return the i-th feature value
     */
    Entry OGetEntry(size_t i) const;

    static bool IsEntryMissing(Entry e) { return e.flag == -1; }
#endif

   private:
    std::vector<Entry> data_;
  };
  /*!
   * \brief get the leaf index
   * \param feat dense feature vector, if the feature is missing the field is
   * set to NaN \return the leaf index of the given feature
   */
  int GetLeafIndex(const FVec& feat) const;
#ifdef __ENCLAVE_OBLIVIOUS__
  /*!
   * \brief get the leaf value obliviously
   * \param feat dense feature vector, if the feature is missing the field is
   * set to NaN \param root_id starting root index of the instance \return the
   * leaf index of the given feature
   */
  bst_float OGetLeafValue(const FVec& feat, unsigned root_id = 0) const;
  bst_float OGetLeafValueCache(const FVec& feat, unsigned root_id = 0) const;
#endif
  /*!
   * \brief calculate the feature contributions
   * (https://arxiv.org/abs/1706.06060) for the tree \param feat dense feature
   * vector, if the feature is missing the field is set to NaN \param
   * out_contribs output vector to hold the contributions \param condition fix
   * one feature to either off (-1) on (1) or not fixed (0 default) \param
   * condition_feature the index of the feature to fix
   */
  void CalculateContributions(const RegTree::FVec& feat,
                              bst_float* out_contribs, int condition = 0,
                              unsigned condition_feature = 0) const;
  /*!
   * \brief Recursive function that computes the feature attributions for a
   * single tree. \param feat dense feature vector, if the feature is missing
   * the field is set to NaN \param phi dense output vector of feature
   * attributions \param node_index the index of the current node in the tree
   * \param unique_depth how many unique features are above the current node in
   * the tree \param parent_unique_path a vector of statistics about our current
   * path through the tree \param parent_zero_fraction what fraction of the
   * parent path weight is coming as 0 (integrated) \param parent_one_fraction
   * what fraction of the parent path weight is coming as 1 (fixed) \param
   * parent_feature_index what feature the parent node used to split \param
   * condition fix one feature to either off (-1) on (1) or not fixed (0
   * default) \param condition_feature the index of the feature to fix \param
   * condition_fraction what fraction of the current weight matches our
   * conditioning feature
   */
  void TreeShap(const RegTree::FVec& feat, bst_float* phi, unsigned node_index,
                unsigned unique_depth, PathElement* parent_unique_path,
                bst_float parent_zero_fraction, bst_float parent_one_fraction,
                int parent_feature_index, int condition,
                unsigned condition_feature, bst_float condition_fraction) const;

  /*!
   * \brief calculate the approximate feature contributions for the given root
   * \param feat dense feature vector, if the feature is missing the field is
   * set to NaN \param out_contribs output vector to hold the contributions
   */
  void CalculateContributionsApprox(const RegTree::FVec& feat,
                                    bst_float* out_contribs) const;
  /*!
   * \brief get next position of the tree given current pid
   * \param pid Current node id.
   * \param fvalue feature value if not missing.
   * \param is_unknown Whether current required feature is missing.
   */
  inline int GetNext(int pid, bst_float fvalue, bool is_unknown) const;
#ifdef __ENCLAVE_OBLIVIOUS__
  /*!
   * \brief get next position of the tree given current pid obliviously
   * \param pid Current node id.
   * \param fvalue feature value if not missing.
   * \param is_unknown Whether current required feature is missing.
   */
  inline int OGetNext(int pid, bst_float fvalue, bool is_unknown) const;
#endif
  /*!
   * \brief dump the model in the requested format as a text string
   * \param fmap feature map that may help give interpretations of feature
   * \param with_stats whether dump out statistics as well
   * \param format the format to dump the model in
   * \return the string of dumped model
   */
  std::string DumpModel(const FeatureMap& fmap, bool with_stats,
                        std::string format) const;
  /*!
   * \brief calculate the mean value for each node, required for feature
   * contributions
   */
  void FillNodeMeanValues();

#ifdef __ENCLAVE_DPOBLIVIOUS__
  /*!
   * \brief get the leaf value dp obliviously
   * \param feat dense feature vector, if the feature is missing the field is
   * set to NaN \param root_id starting root index of the instance \return the
   * leaf index of the given feature
   */
  bst_float DPOGetLeafValue(const FVec& feat, unsigned root_id = 0);

  // void PredictByHistNoCache(DMatrix* p_fmat, std::vector<bst_float>*
  // out_preds, int32_t gid, int32_t num_group,
  //                       RegTree::FVec& feat) ;
  // void PredictByHistLayerNoCache(DMatrix* p_fmat,  std::vector<size_t>&
  // index, size_t layer, RegTree::FVec& p_feat) ; int FirstNodeInLayer(int
  // layer);
  void OPredictByHistNoCache(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                             int32_t gid, int32_t num_group,
                             RegTree::FVec& feat);

  void OPredictByHist(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                      int32_t gid, int32_t num_group, RegTree::FVec& feat);

  struct FVecIndex {
    FVec feat;
    int index;
  };
  void DPOPredictByHist(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                        int32_t gid, int32_t num_group, RegTree::FVec& feat);
  void DPOPredictByHist1(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                         int32_t gid, int32_t num_group, RegTree::FVec& feat);
  void FillOutput(SQueue<FVecIndex>& buffer, std::vector<FVecIndex>* output,
                  int out_size);
  void DPOPredictByHist2(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                         int32_t gid, int32_t num_group, RegTree::FVec& feat);
  void FillOutput2(SQueue<std::vector<RegTree::FVec::Entry>>& buffer, std::vector<FVecIndex>* output,
                  int out_size);
  void DPOPredictByHist3(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                         int32_t gid, int32_t num_group, RegTree::FVec& feat);
  void PagePredict(std::vector<SQueue<std::vector<RegTree::FVec::Entry>>*>& hist, RegTree::FVec& feat, std::vector<bst_float>* out_preds, int32_t gid, int32_t num_group, int page_num, int num_round, int nodes_in_page);

  inline void InitStash() { stash_.InitStash(); };
  class NodeStash {
   private:
    using index_type = uint16_t;
    static const size_t capacity_ = 256;
    static const index_type kInvalidIndex = capacity_;
    xgboost::RegTree::Node stash_[capacity_];
    index_type free_stash[capacity_];
    size_t size_ = 0;
    index_type* tree2stash;
    xgboost::RegTree* tree_;
    float rate = 0;
    bool inited = false;

   public:
    NodeStash(){};
    ~NodeStash() { delete[] tree2stash; };
    inline void SetTree(xgboost::RegTree* tree) {
      tree_ = tree;
      // std::cout<<"setTree nodes:"<<tree->GetNodes().size()<<"
      // size"<<sizeof(tree2stash)/sizeof(tree2stash[0])<<std::endl; for(int
      // i=0;i<tree->GetNodes().size();i++) tree2stash[i]=kInvalidIndex;
      rate = static_cast<float>(capacity_) /
             static_cast<float>(tree->GetNodes().size());
      for (index_type i = 0; i < capacity_; i++) {
        free_stash[i] = i;
      }
    };
    inline void Print() {
      std::cout << "stash inclue::";
      xgboost::RegTree::Node node;
      for (size_t nid = 0; nid < tree_->GetNodes().size(); nid++) {
        std::cout << " tree2stash:" << tree2stash[nid];
        if (Get(nid, &node)) {
          std::cout << " nid:" << nid << " value:" << node.LeafValue();
        }
      }
      std::cout << std::endl;
    }
    inline void InitStash() {
      if (inited) {
        return;
      }
      inited = true;
      tree2stash = new index_type[tree_->GetNodes().size()];
      for (int i = 0; i < tree_->GetNodes().size(); i++)
        tree2stash[i] = kInvalidIndex;
      // std::cout<<"end position!!!!! "<<std::endl;
      for (size_t nid = 0; nid < tree_->GetNodes().size(); nid++) {
        if ((*tree_)[nid].IsLeaf()) {
          float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
          if (r < rate) {
            size_t nid_ = nid;
            while (!(*tree_)[nid_].IsRoot()) {
              //   std::cout << "Set before  nid: " << nid_
              //             << " is Leaf:" << (*tree_)[nid_].IsLeaf()
              //             << " leaf value:" << (*tree_)[nid_].LeafValue()
              //             << std::endl;
              Set(nid_, (*tree_)[nid_]);
              // xgboost::RegTree::Node node;
              // Get(nid_, &node);
              // std::cout << "Set after nid: " << nid_
              //           << " is Leaf:" << node.IsLeaf()
              //           << " leaf value:" << node.LeafValue() << std::endl;
              nid_ = (*tree_)[nid_].Parent();
            }
            Set(nid_, (*tree_)[nid_]);
            // std::cout << "test env........." << std::endl;
            // Print();
          }
        }
      }
      if (false) {
        xgboost::RegTree::Node node;
        for (size_t nid = 0; nid < tree_->GetNodes().size(); nid++) {
          if (Get(nid, &node)) {
            std::cout << "stash nid: " << nid << " is Leaf:" << node.IsLeaf()
                      << "leaf value:" << node.LeafValue() << std::endl;
            std::cout << "tree  nid: " << nid
                      << " is Leaf:" << (*tree_)[nid].IsLeaf()
                      << "leaf value:" << (*tree_)[nid].LeafValue()
                      << std::endl;
          }
        }
      }
    }
    inline index_type Alloc() {
      if (size_ < capacity_) {
        size_++;
        return free_stash[size_ - 1];
      }
      return kInvalidIndex;
    }
    inline bool Set(int nid, xgboost::RegTree::Node& node) {
      if (tree2stash[nid] == kInvalidIndex) {
        tree2stash[nid] = Alloc();
        if (tree2stash[nid] == kInvalidIndex) return false;
      }
      stash_[tree2stash[nid]] = node;
      return true;
    }
    inline bool Get(int nid, xgboost::RegTree::Node* out) {
      if (tree2stash[nid] == kInvalidIndex) {
        return false;
      }
      *out = stash_[tree2stash[nid]];
      return true;
    }
    inline bool GetLeafValue(const xgboost::RegTree::FVec& feat,
                             xgboost::bst_float* out_value) {
      xgboost::bst_node_t nid = 0;
      xgboost::RegTree::Node node;
      while (Get(nid, &node)) {
        // std::cout << "getLeafValue!!! nid: " << nid << " isLeaf:" <<
        // node.IsLeaf() << std::endl;
        if (node.IsLeaf()) {
          *out_value = node.LeafValue();
          // std::cout << "leaf value " << node.LeafValue() << std::endl;
          return true;
        }
        unsigned split_index = node.SplitIndex();
        xgboost::bst_float split_value = node.SplitCond();
        if (feat.IsMissing(split_index)) {
          nid = node.DefaultChild();
        } else {
          if (feat.GetFvalue(split_index) < split_value) {
            nid = node.LeftChild();
          } else {
            nid = node.RightChild();
          }
        }
      }
      return false;
    }
  };
#endif

 private:
  // vector of nodes
  std::vector<Node> nodes_;
  // free node space, used during training process
  std::vector<int> deleted_nodes_;
  // stats of nodes
  std::vector<RTreeNodeStat> stats_;
  std::vector<bst_float> node_mean_values_;
#ifdef __ENCLAVE_DPOBLIVIOUS__
  NodeStash stash_;
#endif
  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  int AllocNode() {
    if (param.num_deleted != 0) {
      int nid = deleted_nodes_.back();
      deleted_nodes_.pop_back();
      nodes_[nid].Reuse();
      --param.num_deleted;
      return nid;
    }
    int nd = param.num_nodes++;
    CHECK_LT(param.num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    return nd;
  }
  // delete a tree node, keep the parent field to allow trace back
  void DeleteNode(int nid) {
    CHECK_GE(nid, 1);
    auto pid = (*this)[nid].Parent();
    if (nid == (*this)[pid].LeftChild()) {
      (*this)[pid].SetLeftChild(kInvalidNodeId);
    } else {
      (*this)[pid].SetRightChild(kInvalidNodeId);
    }

    deleted_nodes_.push_back(nid);
    nodes_[nid].MarkDelete();
    ++param.num_deleted;
  }
  bst_float FillNodeMeanValue(int nid);
};

inline void RegTree::FVec::Init(size_t size) {
  Entry e;
  e.flag = -1;
  data_.resize(size);
  std::fill(data_.begin(), data_.end(), e);
}

inline void RegTree::FVec::Fill(const SparsePage::Inst& inst) {
  for (auto const& entry : inst) {
    if (entry.index >= data_.size()) {
      continue;
    }
    data_[entry.index].fvalue = entry.fvalue;
  }
}

inline void RegTree::FVec::Drop(const SparsePage::Inst& inst) {
  for (auto const& entry : inst) {
    if (entry.index >= data_.size()) {
      continue;
    }
    data_[entry.index].flag = -1;
  }
}

inline size_t RegTree::FVec::Size() const { return data_.size(); }

inline void RegTree::FVec::Print() const {
  for (const Entry& temp : data_) {
    if (temp.flag == -1)
      std::cout << "null ";
    else
      std::cout << temp.fvalue << " ";
  }
  std::cout << std::endl;
}

inline std::vector<RegTree::FVec::Entry>& RegTree::FVec::Data(){
  return data_;
}

inline bst_float RegTree::FVec::GetFvalue(size_t i) const {
  return data_[i].fvalue;
}

inline bool RegTree::FVec::IsMissing(size_t i) const {
  return data_[i].flag == -1;
}

inline int RegTree::GetLeafIndex(const RegTree::FVec& feat) const {
  bst_node_t nid = 0;
  while (!(*this)[nid].IsLeaf()) {
    unsigned split_index = (*this)[nid].SplitIndex();
    nid = this->GetNext(nid, feat.GetFvalue(split_index),
                        feat.IsMissing(split_index));
  }
  return nid;
}

/*! \brief get next position of the tree given current pid */
inline int RegTree::GetNext(int pid, bst_float fvalue, bool is_unknown) const {
  bst_float split_value = (*this)[pid].SplitCond();
  if (is_unknown) {
    return (*this)[pid].DefaultChild();
  } else {
    if (fvalue < split_value) {
      return (*this)[pid].LeftChild();
    } else {
      return (*this)[pid].RightChild();
    }
  }
}

#ifdef __ENCLAVE_OBLIVIOUS__
inline RegTree::FVec::Entry RegTree::FVec::OGetEntry(size_t i) const {
  return ObliviousArrayAccess(data_.data(), i, data_.size());
}

inline bst_float RegTree::OGetLeafValue(const RegTree::FVec& feat,
                                        unsigned root_id) const {
  auto next_id = static_cast<int>(root_id);
  // Need to access every node.
  // Complexity: O(n_tree_nodes * oaccess(feat))
  bst_float sum = 0;
  for (int idx = next_id; idx < this->GetNodes().size(); ++idx) {
    // This is deterministic in oblivious model, i.e. the last layer will be
    // leaf node.
    bool is_leaf = (*this)[idx].IsLeaf();
    // This is deterministic.
    bst_float leaf_value = is_leaf ? (*this)[idx].LeafValue() : 0.0f;
    // We are accessing the node in prediction path.
    bool is_in_path = ObliviousEqual(next_id, idx);
    leaf_value = ObliviousChoose(is_in_path, leaf_value, 0.0f);
    sum += leaf_value;

    unsigned split_index = (*this)[idx].SplitIndex();
    // oaccess to protect the feature to split on.
    auto entry = feat.OGetEntry(split_index);
    // update next node if have not encounter the leaf layer.
    if (!is_leaf) {
      next_id = ObliviousChoose(
          is_in_path,
          OGetNext(idx, entry.fvalue, RegTree::FVec::IsEntryMissing(entry)),
          next_id);
    }
  }
  return sum;
}

inline bst_float RegTree::OGetLeafValueCache(const RegTree::FVec& feat,
                                             unsigned root_id) const {
  auto next_id = static_cast<int>(root_id);
  // Need to access every node.
  // Complexity: O(n_tree_nodes * oaccess(feat))
  int nodes_in_page = 4096 / sizeof(RegTree::Node);
  int npage = this->GetNodes().size() / nodes_in_page;
  bst_float sum = 0;
  int idx = next_id;
  while (idx < this->GetNodes().size()) {
    // This is deterministic in oblivious model, i.e. the last layer will be
    // leaf node.
    bool is_leaf = (*this)[idx].IsLeaf();
    // This is deterministic.
    bst_float leaf_value = is_leaf ? (*this)[idx].LeafValue() : 0.0f;
    // We are accessing the node in prediction path.
    bool is_in_path = ObliviousEqual(next_id, idx);
    leaf_value = ObliviousChoose(is_in_path, leaf_value, 0.0f);
    sum += leaf_value;

    unsigned split_index = (*this)[idx].SplitIndex();
    // oaccess to protect the feature to split on.
    auto entry = feat.OGetEntry(split_index);
    // update next node if have not encounter the leaf layer.
    if (!is_leaf) {
      next_id = ObliviousChoose(
          is_in_path,
          OGetNext(idx, entry.fvalue, RegTree::FVec::IsEntryMissing(entry)),
          next_id);
    }

    int now_page = idx / nodes_in_page;
    int next_page = next_id / nodes_in_page;
    // std::cout<<idx<<" "<<next_id<<std::endl;
    idx = ObliviousChoose(
        now_page < next_page - 1 || (idx >= next_id && now_page <= npage),
        int((now_page + 1.5) * nodes_in_page), next_id);
  }
  return sum;
}

inline int RegTree::OGetNext(int pid, bst_float fvalue, bool is_unknown) const {
  bst_float split_value = (*this)[pid].SplitCond();
  auto next_id =
      ObliviousChoose(ObliviousLess(fvalue, split_value),
                      (*this)[pid].LeftChild(), (*this)[pid].RightChild());
  return ObliviousChoose(is_unknown, (*this)[pid].DefaultChild(), next_id);
}
#endif
#ifdef __ENCLAVE_DPOBLIVIOUS__
/**
 * 以DP-RAM算法为基础实现的dp oblivious算法，面向单个数据推断
 */
inline bst_float RegTree::DPOGetLeafValue(const RegTree::FVec& feat,
                                          unsigned root_id) {
  xgboost::bst_float out_value;
  bool in_stash = stash_.GetLeafValue(feat, &out_value);
  bst_node_t nid = 0;
  while (!(*this)[nid].IsLeaf()) {
    unsigned split_index = (*this)[nid].SplitIndex();
    bst_node_t true_nid = this->GetNext(nid, feat.GetFvalue(split_index),
                                        feat.IsMissing(split_index));
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    bst_node_t random_nid = ObliviousChoose(r < 0.5, (*this)[nid].LeftChild(),
                                            (*this)[nid].RightChild());
    nid = ObliviousChoose(in_stash, random_nid, true_nid);
  }
  return ObliviousChoose(in_stash, out_value, (*this)[nid].LeafValue());
}

/**
 * 节点遍历数据，不使用cache优化，满足oblivious
 */
inline void RegTree::OPredictByHistNoCache(DMatrix* p_fmat,
                                           std::vector<bst_float>* out_preds,
                                           int32_t gid, int32_t num_group,
                                           RegTree::FVec& feat) {
  std::vector<int> index;
  index.resize(p_fmat->Info().num_row_);
  std::fill(index.begin(), index.end(), 0);
  std::vector<bst_float>& preds = *out_preds;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
      auto nsize = batch.Size();
      for (int i = 0; i < nsize; i++) {
        bool is_in_node = index[batch.base_rowid + i] == nid;
        if (!(*this)[nid].IsLeaf()) {
          feat.Fill(batch[i]);
          unsigned split_index = (*this)[nid].SplitIndex();
          auto next = OGetNext(nid, feat.GetFvalue(split_index),
                               feat.IsMissing(split_index));
          index[batch.base_rowid + i] =
              ObliviousChoose(is_in_node, next, index[batch.base_rowid + i]);
          feat.Drop(batch[i]);
        } else {
          auto leaf_value =
              ObliviousChoose(is_in_node, (*this)[nid].LeafValue(), 0.0f);
          preds[(batch.base_rowid + i) * num_group + gid] += leaf_value;
        }
      }
    }
  }
}

/**
 * 节点遍历数据，使用cache优化，满足oblivious
 */
inline void RegTree::OPredictByHist(DMatrix* p_fmat,
                                    std::vector<bst_float>* out_preds,
                                    int32_t gid, int32_t num_group,
                                    RegTree::FVec& feat) {
  std::vector<int> index;
  index.resize(p_fmat->Info().num_row_);
  std::fill(index.begin(), index.end(), 0);

  std::vector<bool> preded;
  preded.resize(p_fmat->Info().num_row_);
  std::fill(preded.begin(), preded.end(), false);

  std::vector<bst_float>& preds = *out_preds;
  int num_nodes_page = 4096 / sizeof(Node);
  for (int npage = 0; npage <= this->GetNodes().size() / num_nodes_page;
       npage++) {
    for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
      auto nsize = batch.Size();
      for (int i = 0; i < nsize; i++) {
        if (npage == 0) {
          while (index[batch.base_rowid + i] >= npage * num_nodes_page &&
                 index[batch.base_rowid + i] < (npage + 1) * num_nodes_page &&
                 index[batch.base_rowid + i] < this->GetNodes().size()) {
            feat.Fill(batch[i]);
            bool is_leaf = (*this)[index[batch.base_rowid + i]].IsLeaf();
            auto leaf_value = ObliviousChoose(
                is_leaf, (*this)[index[batch.base_rowid + i]].LeafValue(),
                0.0f);
            preds[(batch.base_rowid + i) * num_group + gid] += leaf_value;

            preded[batch.base_rowid + i] = ObliviousChoose(
                is_leaf, true, bool(preded[batch.base_rowid + i]));
            unsigned split_index = ObliviousChoose(
                preded[batch.base_rowid + i], 0u,
                (*this)[index[batch.base_rowid + i]].SplitIndex());
            auto next = OGetNext(index[batch.base_rowid + i],
                                 feat.GetFvalue(split_index),
                                 feat.IsMissing(split_index));

            index[batch.base_rowid + i] =
                ObliviousChoose(preded[batch.base_rowid + i],
                                index[batch.base_rowid + i] * 2 + 1, next);
            feat.Drop(batch[i]);
          }
        } else {
          bool in_page =
              index[batch.base_rowid + i] >= npage * num_nodes_page &&
              index[batch.base_rowid + i] < (npage + 1) * num_nodes_page &&
              index[batch.base_rowid + i] < this->GetNodes().size();
          int nid = ObliviousChoose(in_page, int(index[batch.base_rowid + i]),
                                    npage * num_nodes_page);
          feat.Fill(batch[i]);
          bool is_leaf = (*this)[nid].IsLeaf();
          auto leaf_value = ObliviousChoose(is_leaf && in_page,
                                            (*this)[nid].LeafValue(), 0.0f);
          preds[(batch.base_rowid + i) * num_group + gid] += leaf_value;

          preded[batch.base_rowid + i] = ObliviousChoose(
              is_leaf && in_page, true, bool(preded[batch.base_rowid + i]));
          unsigned split_index =
              ObliviousChoose(preded[batch.base_rowid + i] || !in_page, 0u,
                              (*this)[nid].SplitIndex());
          auto next = OGetNext(nid, feat.GetFvalue(split_index),
                               feat.IsMissing(split_index));

          index[batch.base_rowid + i] = ObliviousChoose(
              in_page,
              ObliviousChoose(preded[batch.base_rowid + i],
                              index[batch.base_rowid + i] * 2 + 1, next),
              index[batch.base_rowid + i]);
          feat.Drop(batch[i]);
        }
      }
    }
  }
}

/**
 * 直方图算法，依次按节点将数据分成左右两部分，无private
 * memory，无dummy，non-oblivious
 */
inline void RegTree::DPOPredictByHist(DMatrix* p_fmat,
                                      std::vector<bst_float>* out_preds,
                                      int32_t gid, int32_t num_group,
                                      RegTree::FVec& feat) {
  std::vector<std::vector<FVecIndex>*> hist;
  hist.resize(this->GetNodes().size());
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new std::vector<FVecIndex>;
  }

  int num_nodes_page = 4096 / sizeof(Node);
  // std::cout<<"node num:"<<num_nodes_page<<" node
  // size:"<<sizeof(Node)<<std::endl;
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    for (int i = 0; i < nsize; i++) {
      FVecIndex entry;
      auto& p_feat = entry.feat;
      p_feat.Init(feat.Size());
      p_feat.Fill(batch[i]);
      int nid = 0;
      while (nid >= 0 && nid < num_nodes_page && !(*this)[nid].IsLeaf()) {
        unsigned split_index = (*this)[nid].SplitIndex();
        auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                            p_feat.IsMissing(split_index));
        nid = next;
      }
      entry.index = batch.base_rowid + i;
      hist[nid]->push_back(std::move(entry));
    }
  }
  // for(size_t npage=0; npage<=this->GetNodes().size()/num_nodes_page;npage++){
  //   OPredictByHistPage(p_fmat, index, npage, num_nodes_page,
  //   p_thread_temp[0]);
  // }
  std::vector<bst_float>& preds = *out_preds;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    if ((*this)[nid].IsLeaf()) {
      for (auto& entry : *hist[nid]) {
        preds[entry.index * num_group + gid] += (*this)[nid].LeafValue();
      }
    } else {
      while (!hist[nid]->empty()) {
        FVecIndex& entry = hist[nid]->back();
        auto& p_feat = entry.feat;
        unsigned split_index = (*this)[nid].SplitIndex();
        auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                            p_feat.IsMissing(split_index));
        hist[next]->push_back(std::move(entry));
        hist[nid]->pop_back();
      }
    }
    std::vector<FVecIndex>().swap(*hist[nid]);
  }
}

/**
 * 直方图算法，依次按节点将数据分成左右两部分，添加private
 * memory，添加dummy，满足dp oblivious
 */
inline void RegTree::DPOPredictByHist1(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_preds,
                                       int32_t gid, int32_t num_group,
                                       RegTree::FVec& feat) {
  std::vector<std::vector<FVecIndex>*> hist;
  hist.resize(this->GetNodes().size());
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new std::vector<FVecIndex>;
  }

  int num_nodes_page = 4096 / sizeof(Node);
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    std::cout<<"memcost:"<<batch.MemCostBytes()<<" size:"<<nsize<<" mem per sample:"<<batch.MemCostBytes()/nsize<<std::endl;
    for (int i = 0; i < nsize; i++) {
      FVecIndex entry;
      auto& p_feat = entry.feat;
      p_feat.Init(feat.Size());
      p_feat.Fill(batch[i]);
      entry.index = batch.base_rowid + i;
      hist[0]->push_back(std::move(entry));
    }
  }
  std::vector<bst_float>& preds = *out_preds;
  SQueue<FVecIndex> lbuf{25};
  SQueue<FVecIndex> rbuf{25};
  int num_round = lbuf.capacity() / 4;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    if ((*this)[nid].IsLeaf()) {
      bst_float leaf_value = (*this)[nid].LeafValue();
      for (auto& entry : *hist[nid]) {
        bst_float leaf_value_temp =
            ObliviousChoose(entry.index < 0, 0.0f, leaf_value);
        int index = ObliviousChoose(entry.index < 0, -entry.index, entry.index);
        preds[index * num_group + gid] += leaf_value_temp;
      }
    } else {
      int prefix_sum_l = 0;
      int prefix_sum_r = 0;
      for (int round_base = 0; round_base < hist[nid]->size();
           round_base += num_round) {
        prefix_sum_l -= lbuf.size();
        prefix_sum_r -= rbuf.size();
        int s = std::min(num_round, int(hist[nid]->size() - round_base));
        for (int i = 0; i < s; i++) {
          FVecIndex& entry = hist[nid]->at(round_base + i);
          auto& p_feat = entry.feat;
          unsigned split_index = (*this)[nid].SplitIndex();
          auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                              p_feat.IsMissing(split_index));

          lbuf.push_back(entry, (next == (*this)[nid].LeftChild()));
          rbuf.push_back(entry, (next == (*this)[nid].RightChild()));
        }
        prefix_sum_l += lbuf.size();
        prefix_sum_r += rbuf.size();

        FillOutput(lbuf, hist[(*this)[nid].LeftChild()],
                   DPPrefixSum(prefix_sum_l, num_round) - num_round);
        FillOutput(rbuf, hist[(*this)[nid].RightChild()],
                   DPPrefixSum(prefix_sum_r, num_round) - num_round);
      }
      FillOutput(lbuf, hist[(*this)[nid].LeftChild()],
                 DPPrefixSum(prefix_sum_l, num_round) + num_round);
      FillOutput(rbuf, hist[(*this)[nid].RightChild()],
                 DPPrefixSum(prefix_sum_r, num_round) + num_round);
    }
    std::vector<FVecIndex>().swap(*hist[nid]);
  }
}

inline void RegTree::FillOutput(SQueue<FVecIndex>& buffer,
                                std::vector<FVecIndex>* output, int out_size) {
  while (int(output->size()) < out_size) {
    FVecIndex temp;
    auto real = buffer.pop_font(temp);
    temp.index = ObliviousChoose(real, temp.index, -1);
    output->push_back(std::move(temp));
  }
}
/**
 * 直方图算法，依次按节点将数据分成左右两部分，添加private
 * memory，添加dummy，满足dp oblivious
 * 改进隐私内存，存储样本的每一项。
 */
inline void RegTree::DPOPredictByHist2(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_preds,
                                       int32_t gid, int32_t num_group,
                                       RegTree::FVec& feat) {
  std::vector<std::vector<FVecIndex>*> hist;
  hist.resize(this->GetNodes().size());
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new std::vector<FVecIndex>;
  }

  int num_nodes_page = 4096 / sizeof(Node);
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    for (int i = 0; i < nsize; i++) {
      FVecIndex entry;
      auto& p_feat = entry.feat;
      p_feat.Init(feat.Size());
      p_feat.Fill(batch[i]);
      entry.index = batch.base_rowid + i;
      hist[0]->push_back(std::move(entry));
    }
  }
  std::vector<bst_float>& preds = *out_preds;
  int vec_size = feat.Size();
  SQueue<std::vector<RegTree::FVec::Entry>> lbuf{vec_size};
  SQueue<std::vector<RegTree::FVec::Entry>> rbuf{vec_size};
  int num_round = lbuf.capacity() / 4;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    if ((*this)[nid].IsLeaf()) {
      bst_float leaf_value = (*this)[nid].LeafValue();
      for (auto& entry : *hist[nid]) {
        bst_float leaf_value_temp =
            ObliviousChoose(entry.index < 0, 0.0f, leaf_value);
        int index = ObliviousChoose(entry.index < 0, -entry.index, entry.index);
        preds[index * num_group + gid] += leaf_value_temp;
      }
    } else {
      int prefix_sum_l = 0;
      int prefix_sum_r = 0;
      for (int round_base = 0; round_base < hist[nid]->size();
           round_base += num_round) {
        prefix_sum_l -= lbuf.size();
        prefix_sum_r -= rbuf.size();
        int s = std::min(num_round, int(hist[nid]->size() - round_base));
        for (int i = 0; i < s; i++) {
          FVecIndex& entry = hist[nid]->at(round_base + i);
          auto& p_feat = entry.feat;
          unsigned split_index = (*this)[nid].SplitIndex();
          auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                              p_feat.IsMissing(split_index));

          lbuf.push_back(entry.feat.Data(), entry.index, (next == (*this)[nid].LeftChild()));
          rbuf.push_back(entry.feat.Data(), entry.index, (next == (*this)[nid].RightChild()));
        }
        prefix_sum_l += lbuf.size();
        prefix_sum_r += rbuf.size();

        FillOutput2(lbuf, hist[(*this)[nid].LeftChild()],
                   DPPrefixSum(prefix_sum_l, num_round) - num_round);
        FillOutput2(rbuf, hist[(*this)[nid].RightChild()],
                   DPPrefixSum(prefix_sum_r, num_round) - num_round);
      }
      FillOutput2(lbuf, hist[(*this)[nid].LeftChild()],
                 DPPrefixSum(prefix_sum_l, num_round) + num_round);
      FillOutput2(rbuf, hist[(*this)[nid].RightChild()],
                 DPPrefixSum(prefix_sum_r, num_round) + num_round);
    }
    std::vector<FVecIndex>().swap(*hist[nid]);
  }
}

inline void RegTree::FillOutput2(SQueue<std::vector<RegTree::FVec::Entry>>& buffer,
                                std::vector<FVecIndex>* output, int out_size) {
  while (int(output->size()) < out_size) {
    FVecIndex temp;
    temp.feat.Init(buffer.vecSize());
    auto real = buffer.pop_font(temp.feat.Data(), temp.index);
    temp.index = ObliviousChoose(real, temp.index, -1);
    output->push_back(std::move(temp));
  }
}


inline int LeftPage(int page_num, int nodes_in_page){
  return (page_num*nodes_in_page*2+1)/nodes_in_page;
}

inline int RightPage(int page_num, int nodes_in_page){
  return (((page_num+1)*nodes_in_page-1)*2+2)/nodes_in_page;
}
/**
 * 直方图算法，依次按节点将数据分成左右两部分，添加private
 * memory，添加dummy，满足dp oblivious
 * 改进隐私内存，存储样本的每一项。
 * 改进推断过程，减少输出导致的读写操作。
 * 增加cache优化
 * 
 * 目前的推断结果有误！！！
 */
inline void RegTree::DPOPredictByHist3(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_preds,
                                       int32_t gid, int32_t num_group,
                                       RegTree::FVec& feat) {
  int nodes_in_page = 4;
  // int nodes_in_page = cache_size/sizeof(RegTree::Node);
  int npages = this->GetNodes().size()/nodes_in_page + 1;
  std::vector<SQueue<std::vector<RegTree::FVec::Entry>>*> hist;
  hist.resize(npages, nullptr);
  // std::cout<<"npages:"<<npages<<std::endl;
  int vec_size = feat.Size();
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new SQueue<std::vector<RegTree::FVec::Entry>>(vec_size,6);
  }
  std::vector<bst_float>& preds = *out_preds;
  int num_round = hist[0]->capacity()/3;
  // std::cout<<"num_round:"<<num_round<<std::endl;

  // std::cout<<"break point"<<std::endl;
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    for (int round_base = 0; round_base < nsize; round_base+=num_round) {
      int s = std::min(num_round, int(nsize - round_base));
      // std::cout<<"s:"<<s<<std::endl;
      for (int i = 0; i < s; i++) {
        int index = batch.base_rowid + round_base + i;
        feat.Fill(batch[round_base+i]);
        int nid = 0;
        while(nid<nodes_in_page){
          unsigned split_index = (*this)[nid].SplitIndex();
          nid = GetNext(nid, feat.GetFvalue(split_index), feat.IsMissing(split_index));
          if((*this)[nid].IsLeaf()){
            (*out_preds)[index * num_group + gid] += (*this)[nid].LeafValue();
            break;
          }
        }
        // int page_num_next = nid/nodes_in_page;
        for(int page_num_next=std::max(LeftPage(0, nodes_in_page),1); page_num_next<=RightPage(0, nodes_in_page); page_num_next++){
          if(page_num_next<hist.size()&&!(*this)[nid].IsLeaf()){
            // std::cout<<"page num: "<<page_num_next<<" size:"<<hist[page_num_next]->size()<<std::endl;
            hist[page_num_next]->push_back(feat.Data(), index, nid, page_num_next==int(nid/nodes_in_page));
          }
        }
        feat.Drop(batch[round_base+i]);
      }
      PagePredict(hist, feat, out_preds, gid, num_group, 3, num_round, nodes_in_page);
      PagePredict(hist, feat, out_preds, gid, num_group, 2, num_round, nodes_in_page);
      PagePredict(hist, feat, out_preds, gid, num_group, 1, num_round, nodes_in_page);
      PagePredict(hist, feat, out_preds, gid, num_group, 0, num_round, nodes_in_page);
    }
  }
  for (int i = 0; i < hist.size(); i++)
  {
    PagePredict(hist, feat, out_preds, gid, num_group, i, hist[i]->size(), nodes_in_page);
  }
  // for(int i=0; i<hist.size();i++){
  //   std::cout<<"page:"<<i<<" size:"<<hist[i]->size()<<std::endl;
  // }
}

inline void RegTree::PagePredict(std::vector<SQueue<std::vector<RegTree::FVec::Entry>>*>& hist, RegTree::FVec& feat, std::vector<bst_float>* out_preds, int32_t gid, int32_t num_group, int page_num, int num_round, int nodes_in_page){
  if(hist.size()<=page_num)
    return;
  if(hist[page_num]->size()<num_round)
    return;
  // std::cout<<"page0 size: "<<hist[0]->size()<<std::endl;
  // std::cout<<"page1 size: "<<hist[1]->size()<<std::endl;
  // std::cout<<"pagePredict: "<<page_num<<std::endl;
  for(int i=0;i<num_round;i++){
    int index;
    int nid;
    hist[page_num]->pop_font(feat.Data(), index, nid);
    unsigned split_index = (*this)[nid].SplitIndex();
    auto next = GetNext(nid, feat.GetFvalue(split_index), feat.IsMissing(split_index));
    if((*this)[next].IsLeaf()){
      (*out_preds)[index * num_group + gid] += (*this)[next].LeafValue();
    }else{
      for(int page_num_next=std::max(LeftPage(page_num, nodes_in_page),page_num+1); page_num_next<=RightPage(page_num, nodes_in_page); page_num_next++){
        if(page_num_next<hist.size())
          hist[page_num_next]->push_back(feat.Data(), index, next, page_num_next==int(next/nodes_in_page));
      }
    }
  }
  for(int page_num_next=std::max(LeftPage(page_num, nodes_in_page),page_num+1); page_num_next<=RightPage(page_num, nodes_in_page); page_num_next++){
    PagePredict(hist, feat, out_preds, gid, num_group, page_num_next, num_round, nodes_in_page);
  }
}

#endif
}  // namespace xgboost
#endif  // XGBOOST_TREE_MODEL_H_
