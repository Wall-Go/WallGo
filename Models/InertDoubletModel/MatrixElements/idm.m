(* ::Package:: *)

(*Quit[];*)


If[$InputFileName=="",
	SetDirectory[NotebookDirectory[]],
	SetDirectory[DirectoryName[$InputFileName]]
];
(*Put this if you want to create multiple model-files with the same kernel*)
$GroupMathMultipleModels=True;
$LoadGroupMath=True;
Check[
    Get["WallGoMatrix`"],
    Message[Get::noopen, "WallGoMatrix` at "<>ToString[$UserBaseDirectory]<>"/Applications"];
    Abort[];
]


(* ::Chapter:: *)
(*Inert doublet model*)


(*See 2211.13142 for implementation details -- note our different normalization in the
quartic couplings*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2"};
RepAdjoint={{1,1},{2},0};
HiggsDoublet1={{{0,0},{1}},"C"};
HiggsDoublet2={{{0,0},{1}},"C"};
RepScalar={HiggsDoublet1,HiggsDoublet2};
CouplingName={g3,gw};


Rep1={{{1,0},{1}},"L"};
Rep2={{{1,0},{0}},"R"};
Rep3={{{1,0},{0}},"R"};
Rep4={{{0,0},{1}},"L"};
Rep5={{{0,0},{0}},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5};


(* ::Text:: *)
(*The input for the gauge interactions to DRalgo are then given by*)


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The first element is the vector self-interaction matrix:*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


InputInv={{1,1},{True,False}};
MassTerm1=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,2},{True,False}};
MassTerm2=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{1,2},{True,False}};
MassTerm3=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,1},{True,False}};
MassTerm4=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;


VMass=(
	+m1*MassTerm1
	+m2*MassTerm2
	);


\[Mu]ij=GradMass[VMass[[1]]]//Simplify//SparseArray;


QuarticTerm1=MassTerm1[[1]]^2;
QuarticTerm2=MassTerm2[[1]]^2;
QuarticTerm3=MassTerm1[[1]]*MassTerm2[[1]];
QuarticTerm4=MassTerm3[[1]]*MassTerm4[[1]];
QuarticTerm5=(MassTerm3[[1]]^2+MassTerm4[[1]]^2)//Simplify;


VQuartic=(
	+lam1H*QuarticTerm1
	+lam2H*QuarticTerm2
	+lam3H*QuarticTerm3
	+lam4H*QuarticTerm4
	+lam5H/2*QuarticTerm5
	);


\[Lambda]4=GradQuartic[VQuartic];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet1=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;


Ysff=-GradYukawa[yt1*YukawaDoublet1[[1]]];


YsffC=SparseArray[Simplify[Conjugate[Ysff]//Normal,Assumptions->{yt1>0}]];


ImportModel[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];


(* ::Section:: *)
(*MatrixElements*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermoon
*)


(* ::Subsection:: *)
(*TopL, TopR*)


vev={0,v,0,0,0,0,0,0};
(*using this multiple times, should not changre the outcome -> needs fixing *)
SymmetryBreaking[vev,VevDependentCouplings->True] (*uncomment if you want vev-dependent couplings*)
(*SymmetryBreaking[vev]*)


(*Third generation of fermions*)
ReptL=CreateParticle[{{1,1}},"F"];
ReptR=CreateParticle[{{2,1}},"F"];
(*We group the two left-handed and right-handed bottom together*)
Repb = CreateParticle[{{1,2},3},"F"];


(*Light quarks*)


RepLightQ = CreateParticle[{6,7,8,11,12,13},"F"];


(*left-handed leptons*)
RepLepL = CreateParticle[{4,9,14},"F"];

(*right-handed leptons -- these don't contribute*)
RepLepR = CreateParticle[{5,10,15},"F"];


(*Vector bosons*)
RepGluon=CreateParticle[{1},"V"]; (*Gluons*)
RepW=CreateParticle[{{2,1}},"V"]; (*SU2 gauge bosons*)


(*Scalars bosons*)
RepHiggsh=CreateParticle[{{1,2}},"S"]; (*Higgs*)
RepGoldstoneGpR={{1},"S"}; (*real charged Goldstone*)
RepGoldstoneGpI={{3},"S"}; (*imag charged Golstone*)
RepGoldstoneGp0={{4},"S"}; (*neutral Goldstone*)
RepHiggsH=CreateParticle[{{2,2}},"S"]; (*CP-even inert scalar*)
RepGoldstoneA=CreateParticle[{{2,3},{2,1}},"S"]; (*CP-odd inert and charged scalars.
Note that when lambda4 = lambda5, they have the same mass*)


(*Defining various masses and couplings*)
VectorMass=Join[
	Table[mg2,{i,1,RepGluon[[1]]//Length}],
	Table[mw2,{i,1,RepW[[1]]//Length}]
	];
(*First we give all the fermions the same mass*)
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*Now we replace the entries with the lefthanded lepton indices by the lepton mass*)
(*We don't care about the right-handed leptons, because they don't appear in the diagrams*)
FermionMass[[RepLepL[[1]]]]=ml2;
ScalarMass={mG2,mh2,mG2,mG2,mA2,mH2,mA2,mA2};
ParticleMasses={VectorMass,FermionMass,ScalarMass};

UserMasses={mq2,mg2,mw2,mG2,mh2,mH2,mA2};
UserCouplings=Variables@Normal@{Ysff,gvss,gvff,gvvv,\[Lambda]4,\[Lambda]3,vev}//DeleteDuplicates


ParticleList={
	ReptL,ReptR,Repb,RepLightQ,RepLepL,RepLepR,
	RepGluon,RepW,
	RepHiggsh,RepGoldstoneGp0,RepGoldstoneGpR,RepGoldstoneGpI,
	RepHiggsH,RepGoldstoneA};
ParticleName={
	"TopL","TopR","Bot","LightQuark","LepL","LepR",
	"Gluon","W",
	"Higgs","GoldstoneG0","GoldstoneGpR","GoldstoneGpI",
	"H","A"};


(*
	output of matrix elements
*)
OutputFile="matrixElements.idm";

MatrixElements=ExportMatrixElements[
	OutputFile,
	ParticleList,
	UserMasses,
	UserCouplings,
	ParticleName,
	ParticleMasses,
	{
		TruncateAtLeadingLog->True,
		Replacements->{lam1H->0,lam2H->0,lam4H->0,lam5H->0},
		Format->{"json","txt"},
		NormalizeWithDOF->False}];


(* ::Chapter:: *)
(*Just the standard model contributions*)


(*See 2211.13142 for implementation details -- note our different normalization in the
quartic couplings*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2"};
RepAdjoint={{1,1},{2},0};
HiggsDoublet1={{{0,0},{1}},"C"};
RepScalar={HiggsDoublet1};
CouplingName={g3,gw};


Rep1={{{1,0},{1}},"L"};
Rep2={{{1,0},{0}},"R"};
Rep3={{{1,0},{0}},"R"};
Rep4={{{0,0},{1}},"L"};
Rep5={{{0,0},{0}},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5};


(* ::Text:: *)
(*The input for the gauge interactions to DRalgo are then given by*)


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The first element is the vector self-interaction matrix:*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


InputInv={{1,1},{True,False}};
MassTerm1=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;


VMass=(
	+m1*MassTerm1
	);


\[Mu]ij=GradMass[VMass[[1]]]//Simplify//SparseArray;


QuarticTerm1=MassTerm1[[1]]^2;


VQuartic=(
	+lam1H*QuarticTerm1
	);


\[Lambda]4=GradQuartic[VQuartic];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet1=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;


Ysff=-GradYukawa[yt1*YukawaDoublet1[[1]]];


YsffC=SparseArray[Simplify[Conjugate[Ysff]//Normal,Assumptions->{yt1>0}]];


ImportModel[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];


(* ::Section:: *)
(*MatrixElements*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermoon
*)


vev={0,v,0,0};
(*using this multiple times, should not changre the outcome -> needs fixing *)
SymmetryBreaking[vev,VevDependentCouplings->True] (*uncomment if you want vev-dependent couplings*)
(*SymmetryBreaking[vev]*)


(*Third generation of fermions*)
ReptL=CreateParticle[{{1,1}},"F"];
ReptR=CreateParticle[{{2,1}},"F"];
(*We group the two left-handed and right-handed bottom together*)
Repb = CreateParticle[{{1,2},3},"F"];


(*Light quarks*)


RepLightQ = CreateParticle[{6,7,8,11,12,13},"F"];


(*left-handed leptons*)
RepLepL = CreateParticle[{4,9,14},"F"];

(*right-handed leptons -- these don't contribute*)
RepLepR = CreateParticle[{5,10,15},"F"];


(*Vector bosons*)
RepGluon=CreateParticle[{1},"V"]; (*Gluons*)
RepW=CreateParticle[{{2,1}},"V"]; (*SU2 gauge bosons*)


(*Scalars bosons*)
RepHiggsh=CreateParticle[{{1,2}},"S"]; (*Higgs*)
RepGoldstoneGpR={{1},"S"}; (*real charged Goldstone*)
RepGoldstoneGpI={{3},"S"}; (*imag charged Golstone*)
RepGoldstoneGp0={{4},"S"}; (*neutral Goldstone*)


(*Defining various masses and couplings*)
VectorMass=Join[
	Table[mg2,{i,1,RepGluon[[1]]//Length}],
	Table[mw2,{i,1,RepW[[1]]//Length}]
	];
(*First we give all the fermions the same mass*)
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*Now we replace the entries with the lefthanded lepton indices by the lepton mass*)
(*We don't care about the right-handed leptons, because they don't appear in the diagrams*)
FermionMass[[RepLepL[[1]]]]=ml2;
ScalarMass={mG2,mh2,mG2,mG2};
ParticleMasses={VectorMass,FermionMass,ScalarMass};

UserMasses={mq2,mg2,mw2,mG2,mh2};
UserCouplings=Variables@Normal@{Ysff,gvss,gvff,gvvv,\[Lambda]4,\[Lambda]3,vev}//DeleteDuplicates


ParticleList={
	ReptL,ReptR,Repb,RepLightQ,RepLepL,RepLepR,
	RepGluon,RepW,
	RepHiggsh,RepGoldstoneGp0,RepGoldstoneGpR,RepGoldstoneGpI};
ParticleName={
	"TopL","TopR","Bot","LightQuark","LepL","LepR",
	"Gluon","W",
	"Higgs","GoldstoneG0","GoldstoneGpR","GoldstoneGpI"};


(*
	output of matrix elements
*)

(*Now we have neglected all diagrams involving A and H, to reproduce more closely the matrix elements used in 2211.13142
Note that we will have to add the AA <-> HiggsHiggs and AHiggs <-> HiggsA by hand*)
OutputFile="matrixElements.idmReduced";

MatrixElements=ExportMatrixElements[
	OutputFile,
	ParticleList,
	UserMasses,
	UserCouplings,
	ParticleName,
	ParticleMasses,
	{
		TruncateAtLeadingLog->True,
		Replacements->{lam1H->0},
		Format->{"json","txt"},
		NormalizeWithDOF->False}];






(* ::Chapter:: *)
(*Let's implement the SM as in the SMLightHiggs file*)


(* ::Chapter:: *)
(*QCD+W boson*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2"};
RepAdjoint={{1,1},{2}};
CouplingName={g3,gw};


Rep1={{{1,0},{1}},"L"};
Rep2={{{1,0},{0}},"R"};
Rep3={{{1,0},{0}},"R"};
Rep4={{{0,0},{1}},"L"};
Rep5={{{0,0},{0}},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5};



HiggsDoublet1={{{0,0},{1}},"C"};
HiggsDoublet2={{{0,0},{1}},"C"};
RepScalar={HiggsDoublet1,HiggsDoublet2};


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


InputInv={{1,1},{True,False}};
MassTerm1=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,2},{True,False}};
MassTerm2=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{1,2},{True,False}};
MassTerm3=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,1},{True,False}};
MassTerm4=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;


VMass=(
	+m1*MassTerm1
	+m2*MassTerm2
	);


\[Mu]ij=GradMass[VMass[[1]]]//Simplify//SparseArray;


QuarticTerm1=MassTerm1[[1]]^2;
QuarticTerm2=MassTerm2[[1]]^2;
QuarticTerm3=MassTerm1[[1]]*MassTerm2[[1]];
QuarticTerm4=MassTerm3[[1]]*MassTerm4[[1]];
QuarticTerm5=(MassTerm3[[1]]^2+MassTerm4[[1]]^2)//Simplify;


VQuartic=(
	+lam1H*QuarticTerm1
	+lam2H*QuarticTerm2
	+lam3H*QuarticTerm3
	+lam4H*QuarticTerm4
	+lam5H/2*QuarticTerm5
	);


\[Lambda]4=GradQuartic[VQuartic];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;
Ysff=-GradYukawa[yt1*YukawaDoublet[[1]]];


ImportModel[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];


(* ::Section:: *)
(*SM quarks + gauge bosons + leptons*)


(* ::Subsection:: *)
(*SymmetryBreaking*)


vev={0,v,0,0,0,0,0,0};
SymmetryBreaking[vev]


(* ::Subsection:: *)
(*UserInput*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermoon
*)


(*left-handed top-quark*)
ReptL=CreateParticle[{{1,1}},"F"];

(*right-handed top-quark*)
ReptR=CreateParticle[{{2,1}},"F"];

(*light quarks*)
RepLightQ = CreateParticle[{{1,2},3,6,7,8,11,12,13},"F"];

(*left-handed leptons*)
RepLepL = CreateParticle[{4,9,14},"F"];

(*right-handed leptons -- these don't contribute*)
RepLepR = CreateParticle[{5,10,15},"F"];

(*Vector bosons*)
RepGluon=CreateParticle[{1},"V"];

(*We are approximating the W and the Z as the same particle*)
RepW=CreateParticle[{{2,1}},"V"];

(*Higgs*)
RepHiggs = CreateParticle[{1},"S"];

RepH=CreateParticle[{{2,2}},"S"]; (*CP-even inert scalar*)
RepA=CreateParticle[{{2,3},{2,1}},"S"]; (*CP-odd inert and charged scalars.
Note that when lambda4 = lambda5, they have the same mass*)


(*
These particles do not necessarily have to be out of equilibrium
the remainin particle content is set as light
*)
ParticleList={ReptL,ReptR,RepLightQ,RepLepL,RepLepR,RepGluon,RepW, RepHiggs,RepH, RepA};


(*Defining various masses and couplings*)


VectorMass=Join[
	Table[mg2,{i,1,RepGluon[[1]]//Length}],
	Table[mw2,{i,1,RepW[[1]]//Length}]
	];
(*First we give all the leptons the same mass*)
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*Now we replace the entries with the lefthanded lepton indices by the lepton mass*)
(*We don't care about the right-handed leptons, because they don't appear in the diagrams*)
FermionMass[[RepLepL[[1]]]]=ml2;
ScalarMass={mh2,mh2,mh2,mh2,mA2,mH2,mA2,mA2};
ParticleMasses={VectorMass,FermionMass,ScalarMass};
(*
up to the user to make sure that the same order is given in the python code
*)
UserMasses={mq2,ml2,mg2,mw2,mh2,mA2};
UserCouplings=Variables@Normal@{Ysff,gvss,gvff,gvvv,\[Lambda]4,\[Lambda]3,vev}//DeleteDuplicates


(*
	output of matrix elements
*)
OutputFile="matrixElements.SMA";
SetDirectory[NotebookDirectory[]];
ParticleName={"TopL","TopR","LightQuark","LepL","LepR","Gluon","W","Higgs","H","A"};
MatrixElements=ExportMatrixElements[
	OutputFile,
	ParticleList,
	UserMasses,
	UserCouplings,
	ParticleName,
	ParticleMasses,
	{TruncateAtLeadingLog->True,Replacements->{lam1H->0,lam2H->0,lam4H->0,lam5H->0},Format->{"json","txt"}}];




