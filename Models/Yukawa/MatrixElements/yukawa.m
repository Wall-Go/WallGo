(* ::Package:: *)

(*Quit[];*)


If[$InputFileName=="",
	SetDirectory[NotebookDirectory[]],
	SetDirectory[DirectoryName[$InputFileName]]
];
(*Put this if you want to create multiple model-files with the same kernel*)
WallGo`WallGoMatrix`$GroupMathMultipleModels=True;
WallGo`WallGoMatrix`$LoadGroupMath=True;
Check[
    Get["WallGo`WallGoMatrix`"],
    Message[Get::noopen, "WallGo`WallGoMatrix` at "<>ToString[$UserBaseDirectory]<>"/Applications"];
    Abort[];
]


(* ::Chapter:: *)
(*Yukawa Model*)


(* ::Section:: *)
(*Model*)


Group={"U1"};
RepAdjoint={0};
RepScalar={{{0},"R"}};
CouplingName={g1};


RepFermion={{{0},"L"},{{0},"R"}};


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJC,\[Mu]IJ,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion,RepScalar];


(* \[Sigma] \[Phi] *)
InputInv={{1},{True}};
LinearTerm=CreateInvariant[Group,RepScalar,InputInv][[1]];
VLinear=0;(* turned off *)\[Sigma] LinearTerm
\[Lambda]1=GradTadpole[VLinear];


(* 1/2m^2\[Phi]^2 *)
InputInv={{1,1},{True,True}}; 
MassTerm=CreateInvariant[Group,RepScalar,InputInv][[1]];
VMass=msq/2 MassTerm
\[Mu]ij=GradMass[VMass]//SparseArray;


(* 1/6\[Gamma]\[Phi]^3 *)
InputInv={{1,1,1},{True,True,True}};
CubicTerm=CreateInvariant[Group,RepScalar,InputInv][[1]];
VCubic=gamma/6 CubicTerm
\[Lambda]3=GradCubic[VCubic];


(* 1/24\[Lambda]\[Phi]^4 *)
InputInv={{1,1,1,1},{True,True,True,True}};
QuarticTerm=CreateInvariant[Group,RepScalar,InputInv][[1]];
VQuartic=lam/24 MassTerm^2
\[Lambda]4=GradQuartic[VQuartic];


(* m(Subscript[\[Psi], R]Subscript[\[Psi], L]+Subscript[\[Psi]^+, L]Subscript[\[Psi], R]^+)*)
InputInv={{2,1},{True,True}}; (*Subscript[\[Psi], R]^+Subscript[\[Psi], L]*)
MassTerm1=CreateInvariantFermion[Group,RepFermion,InputInv][[1]]
InputInv={{1,2},{False,False}};  (*Subscript[\[Psi]^+, L]Subscript[\[Psi], R]*)
MassTerm2=CreateInvariantFermion[Group,RepFermion,InputInv][[1]]


\[Mu]IJ=mpsi*GradMassFermion[MassTerm1];
\[Mu]IJC=mpsi*GradMassFermion[MassTerm2];


(* y \[Phi](Subscript[\[Psi], R]Subscript[\[Psi], L]+Subscript[\[Psi]^+, L]Subscript[\[Psi], R]^+)*)
InputInv={{1,2,1},{True,True,True}};
YukawaDoublet1=CreateInvariantYukawa[Group,RepScalar,RepFermion,InputInv][[1]]//Simplify;
InputInv={{1,1,2},{True,False,False}}; 
YukawaDoublet2=CreateInvariantYukawa[Group,RepScalar,RepFermion,InputInv][[1]]//Simplify;


Ysff= y*GradYukawa[YukawaDoublet1];
YsffC=y*GradYukawa[YukawaDoublet2];


ImportModel[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];


(* ::Section:: *)
(*A model with 1 Dirac fermions and 1 scalar bosons*)


(* ::Subsection:: *)
(*SymmetryBreaking*)


gvss//MatrixForm


WallGoMatrix`SimplifySparse


vev={v};
(*SymmetryBreaking[vev,VevDependentCouplings->True]*) (*uncomment if you want vev-dependent couplings*)
SymmetryBreaking[vev]


(* ::Subsection:: *)
(*UserInput*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermion
*)


(*Below
rep 1-2 are fermions,
(*rep 3 is a scalar*)
*)
(* scalar *)
RepScalar=CreateParticle[{1},"S",ms2,"Phi"];

(* left-handed fermion *)
RepFermionL=CreateParticle[{1},"F",mf2,"PsiL"];

(* right-handed fermion *)
RepFermionR=CreateParticle[{2},"F",mf2,"PsiR"];

(*Vector bosons*)
RepZ=CreateParticle[{1},"V",mv2,"Z"];


(*
These particles do not necessarily have to be out of equilibrium
the particle RepZ is set as light
*)
ParticleList={RepScalar,RepFermionL,RepFermionR};
LightParticleList={RepZ};


(*
	output of matrix elements
*)
OutputFile="matrixElements.yukawa";
SetDirectory[NotebookDirectory[]];
MatrixElements=ExportMatrixElements[
	OutputFile,
	ParticleList,
	LightParticleList,
	{
		TruncateAtLeadingLog->True,
		Format->{"json","txt"}}];


MatrixElements
