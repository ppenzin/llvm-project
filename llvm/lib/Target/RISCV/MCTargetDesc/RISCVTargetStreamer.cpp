//===-- RISCVTargetStreamer.cpp - RISC-V Target Streamer Methods ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISC-V specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetStreamer.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/RISCVAttributes.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

using namespace llvm;

// This option controls whether or not we emit ELF attributes for ABI features,
// like RISC-V atomics or X3 usage.
static cl::opt<bool> RiscvAbiAttr(
    "riscv-abi-attributes",
    cl::desc("Enable emitting RISC-V ELF attributes for ABI features"),
    cl::Hidden);

// void createRegOffsetExpression(unsigned Reg,
//                                                  unsigned FrameReg,
//                                                  int64_t Offset,
//                                                  SmallString<64> &CFAExpr) {
//   // Below all the comments about specific CFI instructions and opcodes are
//   // taken directly from DWARF Standard version 5.
//   //
//   // Encode the expression: (Offset + FrameReg) into Expr:
//   SmallString<64> Expr;
//   uint8_t Buffer[16];
//   // Encode  offset:
//   Expr.push_back(dwarf::DW_OP_consts);
//   // The single operand of the DW_OP_consts operation provides a signed
//   // LEB128 integer constant
//   Expr.append(Buffer, Buffer + encodeSLEB128(Offset, Buffer));
//   // Encode  FrameReg:
//   Expr.push_back((uint8_t)dwarf::DW_OP_bregx);
//   // The DW_OP_bregx operation provides the sum of two values specified by
//   its
//   // two operands. The first operand is a register number which is specified
//   by
//   // an unsigned LEB128 number. The second operand is a signed LEB128 offset.
//   Expr.append(Buffer, Buffer + encodeULEB128(FrameReg, Buffer));
//   Expr.push_back(0);
//   // The DW_OP_plus operation pops the top two stack entries, adds them
//   // together, and pushes the result.
//   Expr.push_back((uint8_t)dwarf::DW_OP_plus);
//   // Now pass the encoded Expr to DW_CFA_expression:
//   //
//   // The DW_CFA_expression instruction takes two operands: an unsigned
//   // LEB128 value representing a register number, and a DW_FORM_block value
//   // representing a DWARF expression
//   CFAExpr.push_back(dwarf::DW_CFA_expression);
//   CFAExpr.append(Buffer, Buffer + encodeULEB128(Reg, Buffer));
//   // DW_FORM_block value is unsigned LEB128 length followed by the number of
//   // bytes specified by the length
//   CFAExpr.append(Buffer, Buffer + encodeULEB128(Expr.size(), Buffer));
//   CFAExpr.append(Expr.str());
//   return;
// }

static void appendScalableVectorExpression(const MCRegisterInfo &MRI,
                                           SmallVectorImpl<char> &Expr,
                                           int FixedOffset,
                                           int ScalableOffset) {
  unsigned DwarfVLenB = MRI.getDwarfRegNum(RISCV::VLENB, true);
  uint8_t Buffer[16];
  if (FixedOffset) {
    Expr.push_back(dwarf::DW_OP_consts);
    Expr.append(Buffer, Buffer + encodeSLEB128(FixedOffset, Buffer));
    Expr.push_back((uint8_t)dwarf::DW_OP_plus);
  }

  Expr.push_back((uint8_t)dwarf::DW_OP_consts);
  Expr.append(Buffer, Buffer + encodeSLEB128(ScalableOffset, Buffer));

  Expr.push_back((uint8_t)dwarf::DW_OP_bregx);
  Expr.append(Buffer, Buffer + encodeULEB128(DwarfVLenB, Buffer));
  Expr.push_back(0);

  Expr.push_back((uint8_t)dwarf::DW_OP_mul);
  Expr.push_back((uint8_t)dwarf::DW_OP_plus);
}

RISCVTargetStreamer::RISCVTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {
  MRI = getContext().getRegisterInfo();
}

void RISCVTargetStreamer::finish() { finishAttributeSection(); }
void RISCVTargetStreamer::reset() {}

void RISCVTargetStreamer::emitCFILLVMDefCfaRegScalableOffset(
    int64_t Register, int64_t ScalableOffset, int64_t FixedOffset, SMLoc Loc) {
  SmallString<64> Expr;
  // Build up the expression (Reg + FixedOffset + ScalableOffset * VLENB).
  unsigned DwarfReg = MRI->getDwarfRegNum(Register, true);
  Expr.push_back((uint8_t)(dwarf::DW_OP_breg0 + DwarfReg));
  Expr.push_back(0);

  appendScalableVectorExpression(*MRI, Expr, FixedOffset, ScalableOffset);

  SmallString<64> DefCfaExpr;
  uint8_t Buffer[16];
  DefCfaExpr.push_back(dwarf::DW_CFA_def_cfa_expression);
  DefCfaExpr.append(Buffer, Buffer + encodeULEB128(Expr.size(), Buffer));
  DefCfaExpr.append(Expr.str());

  Streamer.emitCFIEscape(DefCfaExpr.str(), SMLoc());
}

void RISCVTargetStreamer::emitCFILLVMRegAtScalableOffsetFromCfa(
    int64_t Register, int64_t ScalableOffset, int64_t FixedOffset, SMLoc Loc) {
  SmallString<64> Expr;
  unsigned DwarfReg = MRI->getDwarfRegNum(Register, true);
  // Build up the expression (FixedOffset + ScalableOffset * VLENB).
  appendScalableVectorExpression(*MRI, Expr, FixedOffset, ScalableOffset);

  SmallString<64> DefCfaExpr;
  uint8_t Buffer[16];
  DefCfaExpr.push_back(dwarf::DW_CFA_expression);
  DefCfaExpr.append(Buffer, Buffer + encodeULEB128(DwarfReg, Buffer));
  DefCfaExpr.append(Buffer, Buffer + encodeULEB128(Expr.size(), Buffer));
  DefCfaExpr.append(Expr.str());

  Streamer.emitCFIEscape(DefCfaExpr.str(), SMLoc());
}

void RISCVTargetStreamer::emitDirectiveOptionArch(
    ArrayRef<RISCVOptionArchArg> Args) {}
void RISCVTargetStreamer::emitDirectiveOptionExact() {}
void RISCVTargetStreamer::emitDirectiveOptionNoExact() {}
void RISCVTargetStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionPop() {}
void RISCVTargetStreamer::emitDirectiveOptionPush() {}
void RISCVTargetStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {}
void RISCVTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void RISCVTargetStreamer::finishAttributeSection() {}
void RISCVTargetStreamer::emitTextAttribute(unsigned Attribute,
                                            StringRef String) {}
void RISCVTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                               unsigned IntValue,
                                               StringRef StringValue) {}
void RISCVTargetStreamer::setTargetABI(RISCVABI::ABI ABI) {
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialized target ABI");
  TargetABI = ABI;
}

void RISCVTargetStreamer::setFlagsFromFeatures(const MCSubtargetInfo &STI) {
  HasRVC = STI.hasFeature(RISCV::FeatureStdExtC) ||
           STI.hasFeature(RISCV::FeatureStdExtZca);
  HasTSO = STI.hasFeature(RISCV::FeatureStdExtZtso);
}

void RISCVTargetStreamer::emitTargetAttributes(const MCSubtargetInfo &STI,
                                               bool EmitStackAlign) {
  if (EmitStackAlign) {
    unsigned StackAlign;
    if (TargetABI == RISCVABI::ABI_ILP32E)
      StackAlign = 4;
    else if (TargetABI == RISCVABI::ABI_LP64E)
      StackAlign = 8;
    else
      StackAlign = 16;
    emitAttribute(RISCVAttrs::STACK_ALIGN, StackAlign);
  }

  auto ParseResult = RISCVFeatures::parseFeatureBits(
      STI.hasFeature(RISCV::Feature64Bit), STI.getFeatureBits());
  if (!ParseResult) {
    report_fatal_error(ParseResult.takeError());
  } else {
    auto &ISAInfo = *ParseResult;
    emitTextAttribute(RISCVAttrs::ARCH, ISAInfo->toString());
  }

  if (RiscvAbiAttr && STI.hasFeature(RISCV::FeatureStdExtA)) {
    unsigned AtomicABITag;
    if (STI.hasFeature(RISCV::FeatureStdExtZalasr))
      AtomicABITag = static_cast<unsigned>(RISCVAttrs::RISCVAtomicAbiTag::A7);
    else if (STI.hasFeature(RISCV::FeatureNoTrailingSeqCstFence))
      AtomicABITag = static_cast<unsigned>(RISCVAttrs::RISCVAtomicAbiTag::A6C);
    else
      AtomicABITag = static_cast<unsigned>(RISCVAttrs::RISCVAtomicAbiTag::A6S);
    emitAttribute(RISCVAttrs::ATOMIC_ABI, AtomicABITag);
  }
}

// This part is for ascii assembly output
RISCVTargetAsmStreamer::RISCVTargetAsmStreamer(MCStreamer &S,
                                               formatted_raw_ostream &OS)
    : RISCVTargetStreamer(S), OS(OS) {}

void RISCVTargetAsmStreamer::emitDirectiveOptionPush() {
  OS << "\t.option\tpush\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionPop() {
  OS << "\t.option\tpop\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionPIC() {
  OS << "\t.option\tpic\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoPIC() {
  OS << "\t.option\tnopic\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionRVC() {
  OS << "\t.option\trvc\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoRVC() {
  OS << "\t.option\tnorvc\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionExact() {
  OS << "\t.option\texact\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoExact() {
  OS << "\t.option\tnoexact\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionRelax() {
  OS << "\t.option\trelax\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoRelax() {
  OS << "\t.option\tnorelax\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionArch(
    ArrayRef<RISCVOptionArchArg> Args) {
  OS << "\t.option\tarch";
  for (const auto &Arg : Args) {
    OS << ", ";
    switch (Arg.Type) {
    case RISCVOptionArchArgType::Full:
      break;
    case RISCVOptionArchArgType::Plus:
      OS << "+";
      break;
    case RISCVOptionArchArgType::Minus:
      OS << "-";
      break;
    }
    OS << Arg.Value;
  }
  OS << "\n";
}

void RISCVTargetAsmStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  OS << "\t.variant_cc\t" << Symbol.getName() << "\n";
}

void RISCVTargetAsmStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  OS << "\t.attribute\t" << Attribute << ", " << Twine(Value) << "\n";
}

void RISCVTargetAsmStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  OS << "\t.attribute\t" << Attribute << ", \"" << String << "\"\n";
}

void RISCVTargetAsmStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {}

void RISCVTargetAsmStreamer::finishAttributeSection() {}
