; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=slp-vectorizer -S %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes=slp-vectorizer -S %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

; Some negative tests first.

; We need selects with a uniform predicate to lower effectively to vector
; instructions.
define void @select_mixed_predicates_8xi16(ptr %ptr, i16 %x) {
; CHECK-LABEL: @select_mixed_predicates_8xi16(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[L_0:%.*]] = load i16, ptr [[PTR:%.*]], align 2
; CHECK-NEXT:    [[CMP_0:%.*]] = icmp ult i16 [[L_0]], 16383
; CHECK-NEXT:    [[S_0:%.*]] = select i1 [[CMP_0]], i16 [[L_0]], i16 [[X:%.*]]
; CHECK-NEXT:    store i16 [[S_0]], ptr [[PTR]], align 2
; CHECK-NEXT:    [[GEP_1:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 1
; CHECK-NEXT:    [[L_1:%.*]] = load i16, ptr [[GEP_1]], align 2
; CHECK-NEXT:    [[CMP_1:%.*]] = icmp sgt i16 [[L_1]], 16383
; CHECK-NEXT:    [[S_1:%.*]] = select i1 [[CMP_1]], i16 [[L_1]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_1]], ptr [[GEP_1]], align 2
; CHECK-NEXT:    [[GEP_2:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 2
; CHECK-NEXT:    [[L_2:%.*]] = load i16, ptr [[GEP_2]], align 2
; CHECK-NEXT:    [[CMP_2:%.*]] = icmp eq i16 [[L_2]], 16383
; CHECK-NEXT:    [[S_2:%.*]] = select i1 [[CMP_2]], i16 [[L_2]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_2]], ptr [[GEP_2]], align 2
; CHECK-NEXT:    [[GEP_3:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 3
; CHECK-NEXT:    [[L_3:%.*]] = load i16, ptr [[GEP_3]], align 2
; CHECK-NEXT:    [[CMP_3:%.*]] = icmp ne i16 [[L_3]], 16383
; CHECK-NEXT:    [[S_3:%.*]] = select i1 [[CMP_3]], i16 [[L_3]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_3]], ptr [[GEP_3]], align 2
; CHECK-NEXT:    [[GEP_4:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 4
; CHECK-NEXT:    [[L_4:%.*]] = load i16, ptr [[GEP_4]], align 2
; CHECK-NEXT:    [[CMP_4:%.*]] = icmp eq i16 [[L_4]], 16383
; CHECK-NEXT:    [[S_4:%.*]] = select i1 [[CMP_4]], i16 [[L_4]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_4]], ptr [[GEP_4]], align 2
; CHECK-NEXT:    [[GEP_5:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 5
; CHECK-NEXT:    [[L_5:%.*]] = load i16, ptr [[GEP_5]], align 2
; CHECK-NEXT:    [[CMP_5:%.*]] = icmp ule i16 [[L_5]], 16383
; CHECK-NEXT:    [[S_5:%.*]] = select i1 [[CMP_5]], i16 [[L_5]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_5]], ptr [[GEP_5]], align 2
; CHECK-NEXT:    [[GEP_6:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 6
; CHECK-NEXT:    [[L_6:%.*]] = load i16, ptr [[GEP_6]], align 2
; CHECK-NEXT:    [[CMP_6:%.*]] = icmp ult i16 [[L_6]], 16383
; CHECK-NEXT:    [[S_6:%.*]] = select i1 [[CMP_6]], i16 [[L_6]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_6]], ptr [[GEP_6]], align 2
; CHECK-NEXT:    [[GEP_7:%.*]] = getelementptr inbounds i16, ptr [[PTR]], i16 7
; CHECK-NEXT:    [[L_7:%.*]] = load i16, ptr [[GEP_7]], align 2
; CHECK-NEXT:    [[CMP_7:%.*]] = icmp ult i16 [[L_7]], 16383
; CHECK-NEXT:    [[S_7:%.*]] = select i1 [[CMP_7]], i16 [[L_7]], i16 [[X]]
; CHECK-NEXT:    store i16 [[S_7]], ptr [[GEP_7]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i16, ptr %ptr
  %cmp.0 = icmp ult i16 %l.0, 16383
  %s.0 = select i1 %cmp.0, i16 %l.0, i16 %x
  store i16 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i16, ptr %ptr, i16 1
  %l.1 = load i16, ptr %gep.1
  %cmp.1 = icmp sgt i16 %l.1, 16383
  %s.1 = select i1 %cmp.1, i16 %l.1, i16 %x
  store i16 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i16, ptr %ptr, i16 2
  %l.2 = load i16, ptr %gep.2
  %cmp.2 = icmp eq i16 %l.2, 16383
  %s.2 = select i1 %cmp.2, i16 %l.2, i16 %x
  store i16 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i16, ptr %ptr, i16 3
  %l.3 = load i16, ptr %gep.3
  %cmp.3 = icmp ne i16 %l.3, 16383
  %s.3 = select i1 %cmp.3, i16 %l.3, i16 %x
  store i16 %s.3, ptr %gep.3, align 2

  %gep.4 = getelementptr inbounds i16, ptr %ptr, i16 4
  %l.4 = load i16, ptr %gep.4
  %cmp.4 = icmp eq i16 %l.4, 16383
  %s.4 = select i1 %cmp.4, i16 %l.4, i16 %x
  store i16 %s.4, ptr %gep.4, align 2

  %gep.5 = getelementptr inbounds i16, ptr %ptr, i16 5
  %l.5 = load i16, ptr %gep.5
  %cmp.5 = icmp ule i16 %l.5, 16383
  %s.5 = select i1 %cmp.5, i16 %l.5, i16 %x
  store i16 %s.5, ptr %gep.5, align 2

  %gep.6 = getelementptr inbounds i16, ptr %ptr, i16 6
  %l.6 = load i16, ptr %gep.6
  %cmp.6 = icmp ult i16 %l.6, 16383
  %s.6 = select i1 %cmp.6, i16 %l.6, i16 %x
  store i16 %s.6, ptr %gep.6, align 2

  %gep.7 = getelementptr inbounds i16, ptr %ptr, i16 7
  %l.7 = load i16, ptr %gep.7
  %cmp.7 = icmp ult i16 %l.7, 16383
  %s.7 = select i1 %cmp.7, i16 %l.7, i16 %x
  store i16 %s.7, ptr %gep.7, align 2
  ret void
}

define void @select_uniform_ugt_7xi8(ptr %ptr, i8 %x) {
; CHECK-LABEL: @select_uniform_ugt_7xi8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i8>, ptr [[PTR:%.*]], align 1
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt <4 x i8> [[TMP0]], splat (i8 -1)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i8> poison, i8 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <4 x i8> [[TMP2]], <4 x i8> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <4 x i1> [[TMP1]], <4 x i8> [[TMP0]], <4 x i8> [[TMP3]]
; CHECK-NEXT:    store <4 x i8> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    [[GEP_4:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i8 4
; CHECK-NEXT:    [[L_4:%.*]] = load i8, ptr [[GEP_4]], align 1
; CHECK-NEXT:    [[CMP_4:%.*]] = icmp ugt i8 [[L_4]], -1
; CHECK-NEXT:    [[S_4:%.*]] = select i1 [[CMP_4]], i8 [[L_4]], i8 [[X]]
; CHECK-NEXT:    store i8 [[S_4]], ptr [[GEP_4]], align 2
; CHECK-NEXT:    [[GEP_5:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i8 5
; CHECK-NEXT:    [[L_5:%.*]] = load i8, ptr [[GEP_5]], align 1
; CHECK-NEXT:    [[CMP_5:%.*]] = icmp ugt i8 [[L_5]], -1
; CHECK-NEXT:    [[S_5:%.*]] = select i1 [[CMP_5]], i8 [[L_5]], i8 [[X]]
; CHECK-NEXT:    store i8 [[S_5]], ptr [[GEP_5]], align 2
; CHECK-NEXT:    [[GEP_6:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i8 6
; CHECK-NEXT:    [[L_6:%.*]] = load i8, ptr [[GEP_6]], align 1
; CHECK-NEXT:    [[CMP_6:%.*]] = icmp ugt i8 [[L_6]], -1
; CHECK-NEXT:    [[S_6:%.*]] = select i1 [[CMP_6]], i8 [[L_6]], i8 [[X]]
; CHECK-NEXT:    store i8 [[S_6]], ptr [[GEP_6]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i8, ptr %ptr
  %cmp.0 = icmp ugt i8 %l.0, 16383
  %s.0 = select i1 %cmp.0, i8 %l.0, i8 %x
  store i8 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i8, ptr %ptr, i8 1
  %l.1 = load i8, ptr %gep.1
  %cmp.1 = icmp ugt i8 %l.1, 16383
  %s.1 = select i1 %cmp.1, i8 %l.1, i8 %x
  store i8 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i8, ptr %ptr, i8 2
  %l.2 = load i8, ptr %gep.2
  %cmp.2 = icmp ugt i8 %l.2, 16383
  %s.2 = select i1 %cmp.2, i8 %l.2, i8 %x
  store i8 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i8, ptr %ptr, i8 3
  %l.3 = load i8, ptr %gep.3
  %cmp.3 = icmp ugt i8 %l.3, 16383
  %s.3 = select i1 %cmp.3, i8 %l.3, i8 %x
  store i8 %s.3, ptr %gep.3, align 2

  %gep.4 = getelementptr inbounds i8, ptr %ptr, i8 4
  %l.4 = load i8, ptr %gep.4
  %cmp.4 = icmp ugt i8 %l.4, 16383
  %s.4 = select i1 %cmp.4, i8 %l.4, i8 %x
  store i8 %s.4, ptr %gep.4, align 2

  %gep.5 = getelementptr inbounds i8, ptr %ptr, i8 5
  %l.5 = load i8, ptr %gep.5
  %cmp.5 = icmp ugt i8 %l.5, 16383
  %s.5 = select i1 %cmp.5, i8 %l.5, i8 %x
  store i8 %s.5, ptr %gep.5, align 2

  %gep.6 = getelementptr inbounds i8, ptr %ptr, i8 6
  %l.6 = load i8, ptr %gep.6
  %cmp.6 = icmp ugt i8 %l.6, 16383
  %s.6 = select i1 %cmp.6, i8 %l.6, i8 %x
  store i8 %s.6, ptr %gep.6, align 2

  ret void
}


; Positive tests.

define void @select_uniform_ugt_8xi8(ptr %ptr, i8 %x) {
; CHECK-LABEL: @select_uniform_ugt_8xi8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i8>, ptr [[PTR:%.*]], align 1
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt <8 x i8> [[TMP0]], splat (i8 -1)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <8 x i8> poison, i8 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <8 x i8> [[TMP2]], <8 x i8> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <8 x i1> [[TMP1]], <8 x i8> [[TMP0]], <8 x i8> [[TMP3]]
; CHECK-NEXT:    store <8 x i8> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i8, ptr %ptr
  %cmp.0 = icmp ugt i8 %l.0, 16383
  %s.0 = select i1 %cmp.0, i8 %l.0, i8 %x
  store i8 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i8, ptr %ptr, i8 1
  %l.1 = load i8, ptr %gep.1
  %cmp.1 = icmp ugt i8 %l.1, 16383
  %s.1 = select i1 %cmp.1, i8 %l.1, i8 %x
  store i8 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i8, ptr %ptr, i8 2
  %l.2 = load i8, ptr %gep.2
  %cmp.2 = icmp ugt i8 %l.2, 16383
  %s.2 = select i1 %cmp.2, i8 %l.2, i8 %x
  store i8 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i8, ptr %ptr, i8 3
  %l.3 = load i8, ptr %gep.3
  %cmp.3 = icmp ugt i8 %l.3, 16383
  %s.3 = select i1 %cmp.3, i8 %l.3, i8 %x
  store i8 %s.3, ptr %gep.3, align 2

  %gep.4 = getelementptr inbounds i8, ptr %ptr, i8 4
  %l.4 = load i8, ptr %gep.4
  %cmp.4 = icmp ugt i8 %l.4, 16383
  %s.4 = select i1 %cmp.4, i8 %l.4, i8 %x
  store i8 %s.4, ptr %gep.4, align 2

  %gep.5 = getelementptr inbounds i8, ptr %ptr, i8 5
  %l.5 = load i8, ptr %gep.5
  %cmp.5 = icmp ugt i8 %l.5, 16383
  %s.5 = select i1 %cmp.5, i8 %l.5, i8 %x
  store i8 %s.5, ptr %gep.5, align 2

  %gep.6 = getelementptr inbounds i8, ptr %ptr, i8 6
  %l.6 = load i8, ptr %gep.6
  %cmp.6 = icmp ugt i8 %l.6, 16383
  %s.6 = select i1 %cmp.6, i8 %l.6, i8 %x
  store i8 %s.6, ptr %gep.6, align 2

  %gep.7 = getelementptr inbounds i8, ptr %ptr, i8 7
  %l.7 = load i8, ptr %gep.7
  %cmp.7 = icmp ugt i8 %l.7, 16383
  %s.7 = select i1 %cmp.7, i8 %l.7, i8 %x
  store i8 %s.7, ptr %gep.7, align 2
  ret void
}

define void @select_uniform_ugt_16xi8(ptr %ptr, i8 %x) {
; CHECK-LABEL: @select_uniform_ugt_16xi8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP_8:%.*]] = getelementptr inbounds i8, ptr [[PTR:%.*]], i8 8
; CHECK-NEXT:    [[L_8:%.*]] = load i8, ptr [[GEP_8]], align 1
; CHECK-NEXT:    [[CMP_8:%.*]] = icmp ugt i8 [[L_8]], -1
; CHECK-NEXT:    [[GEP_9:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i8 9
; CHECK-NEXT:    [[GEP_11:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i8 11
; CHECK-NEXT:    [[L_11:%.*]] = load i8, ptr [[GEP_11]], align 1
; CHECK-NEXT:    [[GEP_12:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i8 12
; CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i8>, ptr [[PTR]], align 1
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[PTR]], align 1
; CHECK-NEXT:    [[S_8:%.*]] = select i1 [[CMP_8]], i8 [[TMP1]], i8 [[X:%.*]]
; CHECK-NEXT:    [[TMP2:%.*]] = load <2 x i8>, ptr [[GEP_9]], align 1
; CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i8>, ptr [[GEP_12]], align 1
; CHECK-NEXT:    [[TMP7:%.*]] = shufflevector <8 x i8> [[TMP0]], <8 x i8> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:    [[TMP11:%.*]] = shufflevector <2 x i8> [[TMP2]], <2 x i8> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <2 x i8> [[TMP2]], <2 x i8> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:    [[TMP5:%.*]] = shufflevector <8 x i8> [[TMP0]], <8 x i8> [[TMP4]], <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <16 x i8> [[TMP5]], i8 [[L_11]], i32 11
; CHECK-NEXT:    [[TMP15:%.*]] = shufflevector <16 x i8> [[TMP6]], <16 x i8> [[TMP7]], <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[TMP10:%.*]] = shufflevector <4 x i8> [[TMP3]], <4 x i8> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:    [[TMP8:%.*]] = shufflevector <16 x i8> [[TMP15]], <16 x i8> [[TMP10]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ugt <16 x i8> [[TMP8]], splat (i8 -1)
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <16 x i8> poison, i8 [[X]], i32 0
; CHECK-NEXT:    [[TMP13:%.*]] = shufflevector <16 x i8> [[TMP12]], <16 x i8> poison, <16 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP14:%.*]] = select <16 x i1> [[TMP9]], <16 x i8> [[TMP8]], <16 x i8> [[TMP13]]
; CHECK-NEXT:    store <16 x i8> [[TMP14]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i8, ptr %ptr
  %cmp.0 = icmp ugt i8 %l.0, 16383
  %s.0 = select i1 %cmp.0, i8 %l.0, i8 %x
  store i8 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i8, ptr %ptr, i8 1
  %l.1 = load i8, ptr %gep.1
  %cmp.1 = icmp ugt i8 %l.1, 16383
  %s.1 = select i1 %cmp.1, i8 %l.1, i8 %x
  store i8 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i8, ptr %ptr, i8 2
  %l.2 = load i8, ptr %gep.2
  %cmp.2 = icmp ugt i8 %l.2, 16383
  %s.2 = select i1 %cmp.2, i8 %l.2, i8 %x
  store i8 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i8, ptr %ptr, i8 3
  %l.3 = load i8, ptr %gep.3
  %cmp.3 = icmp ugt i8 %l.3, 16383
  %s.3 = select i1 %cmp.3, i8 %l.3, i8 %x
  store i8 %s.3, ptr %gep.3, align 2

  %gep.4 = getelementptr inbounds i8, ptr %ptr, i8 4
  %l.4 = load i8, ptr %gep.4
  %cmp.4 = icmp ugt i8 %l.4, 16383
  %s.4 = select i1 %cmp.4, i8 %l.4, i8 %x
  store i8 %s.4, ptr %gep.4, align 2

  %gep.5 = getelementptr inbounds i8, ptr %ptr, i8 5
  %l.5 = load i8, ptr %gep.5
  %cmp.5 = icmp ugt i8 %l.5, 16383
  %s.5 = select i1 %cmp.5, i8 %l.5, i8 %x
  store i8 %s.5, ptr %gep.5, align 2

  %gep.6 = getelementptr inbounds i8, ptr %ptr, i8 6
  %l.6 = load i8, ptr %gep.6
  %cmp.6 = icmp ugt i8 %l.6, 16383
  %s.6 = select i1 %cmp.6, i8 %l.6, i8 %x
  store i8 %s.6, ptr %gep.6, align 2

  %gep.7 = getelementptr inbounds i8, ptr %ptr, i8 7
  %l.7 = load i8, ptr %gep.7
  %cmp.7 = icmp ugt i8 %l.7, 16383
  %s.7 = select i1 %cmp.7, i8 %l.7, i8 %x
  store i8 %s.7, ptr %gep.7, align 2

  %gep.8 = getelementptr inbounds i8, ptr %ptr, i8 8
  %l.8 = load i8, ptr %gep.8
  %cmp.8 = icmp ugt i8 %l.8, 16383
  %s.8 = select i1 %cmp.8, i8 %l.0, i8 %x
  store i8 %s.0, ptr %gep.8, align 2

  %gep.9 = getelementptr inbounds i8, ptr %ptr, i8 9
  %l.9 = load i8, ptr %gep.9
  %cmp.9 = icmp ugt i8 %l.9, 16383
  %s.9 = select i1 %cmp.9, i8 %l.9, i8 %x
  store i8 %s.9, ptr %gep.9, align 2

  %gep.10 = getelementptr inbounds i8, ptr %ptr, i8 10
  %l.10 = load i8, ptr %gep.10
  %cmp.10 = icmp ugt i8 %l.10, 16383
  %s.10 = select i1 %cmp.10, i8 %l.10, i8 %x
  store i8 %s.10, ptr %gep.10, align 2

  %gep.11 = getelementptr inbounds i8, ptr %ptr, i8 11
  %l.11 = load i8, ptr %gep.11
  %cmp.11 = icmp ugt i8 %l.11, 16383
  %s.11 = select i1 %cmp.11, i8 %l.11, i8 %x
  store i8 %s.11, ptr %gep.11, align 2

  %gep.12 = getelementptr inbounds i8, ptr %ptr, i8 12
  %l.12 = load i8, ptr %gep.12
  %cmp.12 = icmp ugt i8 %l.12, 16383
  %s.12 = select i1 %cmp.12, i8 %l.12, i8 %x
  store i8 %s.12, ptr %gep.12, align 2

  %gep.13 = getelementptr inbounds i8, ptr %ptr, i8 13
  %l.13 = load i8, ptr %gep.13
  %cmp.13 = icmp ugt i8 %l.13, 16383
  %s.13 = select i1 %cmp.13, i8 %l.13, i8 %x
  store i8 %s.13, ptr %gep.13, align 2

  %gep.14 = getelementptr inbounds i8, ptr %ptr, i8 14
  %l.14 = load i8, ptr %gep.14
  %cmp.14 = icmp ugt i8 %l.14, 16383
  %s.14 = select i1 %cmp.14, i8 %l.14, i8 %x
  store i8 %s.14, ptr %gep.14, align 2

  %gep.15 = getelementptr inbounds i8, ptr %ptr, i8 15
  %l.15 = load i8, ptr %gep.15
  %cmp.15 = icmp ugt i8 %l.15, 16383
  %s.15 = select i1 %cmp.15, i8 %l.15, i8 %x
  store i8 %s.15, ptr %gep.15, align 2

  ret void
}


define void @select_uniform_ugt_4xi16(ptr %ptr, i16 %x) {
; CHECK-LABEL: @select_uniform_ugt_4xi16(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i16>, ptr [[PTR:%.*]], align 2
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt <4 x i16> [[TMP0]], splat (i16 16383)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i16> poison, i16 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <4 x i16> [[TMP2]], <4 x i16> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <4 x i1> [[TMP1]], <4 x i16> [[TMP0]], <4 x i16> [[TMP3]]
; CHECK-NEXT:    store <4 x i16> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i16, ptr %ptr
  %cmp.0 = icmp ugt i16 %l.0, 16383
  %s.0 = select i1 %cmp.0, i16 %l.0, i16 %x
  store i16 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i16, ptr %ptr, i16 1
  %l.1 = load i16, ptr %gep.1
  %cmp.1 = icmp ugt i16 %l.1, 16383
  %s.1 = select i1 %cmp.1, i16 %l.1, i16 %x
  store i16 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i16, ptr %ptr, i16 2
  %l.2 = load i16, ptr %gep.2
  %cmp.2 = icmp ugt i16 %l.2, 16383
  %s.2 = select i1 %cmp.2, i16 %l.2, i16 %x
  store i16 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i16, ptr %ptr, i16 3
  %l.3 = load i16, ptr %gep.3
  %cmp.3 = icmp ugt i16 %l.3, 16383
  %s.3 = select i1 %cmp.3, i16 %l.3, i16 %x
  store i16 %s.3, ptr %gep.3, align 2

  ret void
}

define void @select_uniform_ult_8xi16(ptr %ptr, i16 %x) {
; CHECK-LABEL: @select_uniform_ult_8xi16(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i16>, ptr [[PTR:%.*]], align 2
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult <8 x i16> [[TMP0]], splat (i16 16383)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <8 x i16> poison, i16 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <8 x i16> [[TMP2]], <8 x i16> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <8 x i1> [[TMP1]], <8 x i16> [[TMP0]], <8 x i16> [[TMP3]]
; CHECK-NEXT:    store <8 x i16> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i16, ptr %ptr
  %cmp.0 = icmp ult i16 %l.0, 16383
  %s.0 = select i1 %cmp.0, i16 %l.0, i16 %x
  store i16 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i16, ptr %ptr, i16 1
  %l.1 = load i16, ptr %gep.1
  %cmp.1 = icmp ult i16 %l.1, 16383
  %s.1 = select i1 %cmp.1, i16 %l.1, i16 %x
  store i16 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i16, ptr %ptr, i16 2
  %l.2 = load i16, ptr %gep.2
  %cmp.2 = icmp ult i16 %l.2, 16383
  %s.2 = select i1 %cmp.2, i16 %l.2, i16 %x
  store i16 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i16, ptr %ptr, i16 3
  %l.3 = load i16, ptr %gep.3
  %cmp.3 = icmp ult i16 %l.3, 16383
  %s.3 = select i1 %cmp.3, i16 %l.3, i16 %x
  store i16 %s.3, ptr %gep.3, align 2

  %gep.4 = getelementptr inbounds i16, ptr %ptr, i16 4
  %l.4 = load i16, ptr %gep.4
  %cmp.4 = icmp ult i16 %l.4, 16383
  %s.4 = select i1 %cmp.4, i16 %l.4, i16 %x
  store i16 %s.4, ptr %gep.4, align 2

  %gep.5 = getelementptr inbounds i16, ptr %ptr, i16 5
  %l.5 = load i16, ptr %gep.5
  %cmp.5 = icmp ult i16 %l.5, 16383
  %s.5 = select i1 %cmp.5, i16 %l.5, i16 %x
  store i16 %s.5, ptr %gep.5, align 2

  %gep.6 = getelementptr inbounds i16, ptr %ptr, i16 6
  %l.6 = load i16, ptr %gep.6
  %cmp.6 = icmp ult i16 %l.6, 16383
  %s.6 = select i1 %cmp.6, i16 %l.6, i16 %x
  store i16 %s.6, ptr %gep.6, align 2

  %gep.7 = getelementptr inbounds i16, ptr %ptr, i16 7
  %l.7 = load i16, ptr %gep.7
  %cmp.7 = icmp ult i16 %l.7, 16383
  %s.7 = select i1 %cmp.7, i16 %l.7, i16 %x
  store i16 %s.7, ptr %gep.7, align 2
  ret void
}

define void @select_uniform_eq_2xi32(ptr %ptr, i32 %x) {
; CHECK-LABEL: @select_uniform_eq_2xi32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[PTR:%.*]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i32> [[TMP0]], splat (i32 16383)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x i32> poison, i32 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <2 x i1> [[TMP1]], <2 x i32> [[TMP0]], <2 x i32> [[TMP3]]
; CHECK-NEXT:    store <2 x i32> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i32, ptr %ptr
  %cmp.0 = icmp eq i32 %l.0, 16383
  %s.0 = select i1 %cmp.0, i32 %l.0, i32 %x
  store i32 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i32, ptr %ptr, i32 1
  %l.1 = load i32, ptr %gep.1
  %cmp.1 = icmp eq i32 %l.1, 16383
  %s.1 = select i1 %cmp.1, i32 %l.1, i32 %x
  store i32 %s.1, ptr %gep.1, align 2

  ret void
}

define void @select_uniform_eq_4xi32(ptr %ptr, i32 %x) {
; CHECK-LABEL: @select_uniform_eq_4xi32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i32>, ptr [[PTR:%.*]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <4 x i32> [[TMP0]], splat (i32 16383)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <4 x i1> [[TMP1]], <4 x i32> [[TMP0]], <4 x i32> [[TMP3]]
; CHECK-NEXT:    store <4 x i32> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i32, ptr %ptr
  %cmp.0 = icmp eq i32 %l.0, 16383
  %s.0 = select i1 %cmp.0, i32 %l.0, i32 %x
  store i32 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i32, ptr %ptr, i32 1
  %l.1 = load i32, ptr %gep.1
  %cmp.1 = icmp eq i32 %l.1, 16383
  %s.1 = select i1 %cmp.1, i32 %l.1, i32 %x
  store i32 %s.1, ptr %gep.1, align 2

  %gep.2 = getelementptr inbounds i32, ptr %ptr, i32 2
  %l.2 = load i32, ptr %gep.2
  %cmp.2 = icmp eq i32 %l.2, 16383
  %s.2 = select i1 %cmp.2, i32 %l.2, i32 %x
  store i32 %s.2, ptr %gep.2, align 2

  %gep.3 = getelementptr inbounds i32, ptr %ptr, i32 3
  %l.3 = load i32, ptr %gep.3
  %cmp.3 = icmp eq i32 %l.3, 16383
  %s.3 = select i1 %cmp.3, i32 %l.3, i32 %x
  store i32 %s.3, ptr %gep.3, align 2
  ret void
}

define void @select_uniform_ne_2xi64(ptr %ptr, i64 %x) {
; CHECK-LABEL: @select_uniform_ne_2xi64(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i64>, ptr [[PTR:%.*]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ne <2 x i64> [[TMP0]], splat (i64 16383)
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x i64> poison, i64 [[X:%.*]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <2 x i64> [[TMP2]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = select <2 x i1> [[TMP1]], <2 x i64> [[TMP0]], <2 x i64> [[TMP3]]
; CHECK-NEXT:    store <2 x i64> [[TMP4]], ptr [[PTR]], align 2
; CHECK-NEXT:    ret void
;
entry:
  %l.0 = load i64, ptr %ptr
  %cmp.0 = icmp ne i64 %l.0, 16383
  %s.0 = select i1 %cmp.0, i64 %l.0, i64 %x
  store i64 %s.0, ptr %ptr, align 2

  %gep.1 = getelementptr inbounds i64, ptr %ptr, i64 1
  %l.1 = load i64, ptr %gep.1
  %cmp.1 = icmp ne i64 %l.1, 16383
  %s.1 = select i1 %cmp.1, i64 %l.1, i64 %x
  store i64 %s.1, ptr %gep.1, align 2

  ret void
}
