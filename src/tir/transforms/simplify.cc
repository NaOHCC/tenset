/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file simplify.cc
 * \brief Statement simplifier based on analyzer
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tir;

class StmtSimplifier : public IRMutatorWithAnalyzer {
 private:
  bool disable_canonical_simplify_;

 public:
  explicit StmtSimplifier(Analyzer* analyzer, bool disable_canonical_simplify)
      : IRMutatorWithAnalyzer(analyzer), disable_canonical_simplify_(disable_canonical_simplify) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr(const PrimExpr& expr) final { return analyzer_->Simplify(expr); }

  PrimExpr VisitExpr(const PrimExpr& expr, bool disable_rewrite_simplify,
                     bool disable_canonical_simplify) {
    return analyzer_->Simplify(expr, disable_rewrite_simplify, disable_canonical_simplify);
  }

  Stmt Simplify(Stmt stmt) { return operator()(std::move(stmt)); }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  bool CanInlineLetStmt(const LetStmtNode* op) {
    if (is_const_number(op->value)) return true;
    if (op->value.as<VarNode>()) return true;
    // Won't face the deep expression explosion problem as in Let expression.
    // attempt to inline as much as possible if the value integer type(can be index).
    if (!op->value.dtype().is_int()) return false;
    return SideEffect(op->value) <= CallEffectKind::kPure;
  }

  Stmt VisitStmt_(const LetStmtNode* op) {
    PrimExpr value = this->VisitExpr(op->value);
    if (CanInlineLetStmt(op)) {
      // it is fine to discard the let binding
      // because the call to simplify will always inline the var.
      analyzer_->Bind(op->var, value);
      return this->VisitStmt(op->body);
    }
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  // Control simplify store expression.
  Stmt MyVisitStmt_(const StoreNode* op, bool disable_rewrite_simplify,
                    bool disable_canonical_simplify) {
    PrimExpr value =
        this->VisitExpr(op->value, disable_rewrite_simplify, disable_canonical_simplify);
    PrimExpr index =
        this->VisitExpr(op->index, disable_rewrite_simplify, disable_canonical_simplify);
    PrimExpr predicate = this->VisitExpr(op->predicate);
    if (value.same_as(op->value) && index.same_as(op->index) && predicate.same_as(op->predicate)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->value = std::move(value);
      n->index = std::move(index);
      n->predicate = std::move(predicate);
      return Stmt(n);
    }
  }

  // eliminate useless stores
  Stmt VisitStmt_(const StoreNode* op) final {
    // if (!disable_canonical_simplify_) {
    //   Stmt stmt = Parent::VisitStmt_(op);
    //   op = stmt.as<StoreNode>();
    // }
    Stmt stmt = MyVisitStmt_(op, false, disable_canonical_simplify_);
    op = stmt.as<StoreNode>();

    if (const LoadNode* load = op->value.as<LoadNode>()) {
      if (load->buffer_var.same_as(op->buffer_var) &&
          tir::ExprDeepEqual()(load->index, op->index)) {
        return Evaluate(0);
      }
    }
    return GetRef<Stmt>(op);
  }
};

}  // namespace arith

namespace tir {
namespace transform {

Pass Simplify(bool disable_canonical_simplify) {
  auto pass_func = [disable_canonical_simplify](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    arith::Analyzer analyzer;
    n->body =
        arith::StmtSimplifier(&analyzer, disable_canonical_simplify).Simplify(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.Simplify", {});
}

TVM_REGISTER_GLOBAL("tir.transform.Simplify").set_body_typed(Simplify);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
