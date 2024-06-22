(* ::Package:: *)

(*Quit[];*)


SetDirectory[DirectoryName[$InputFileName]];
(*Put this if you want to create multiple model-files with the same kernel*)
$GroupMathMultipleModels=True;
$LoadGroupMath=True;
<<DRalgo`;
<<matrixElements`;


(* ::Chapter:: *)
(*QCD+W boson*)


$UserBaseDirectory


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2"};
RepAdjoint={{1,1},{2}};
CouplingName={gs,gw};


Rep1={{{1,0},{1}},"L"};
Rep2={{{1,0},{0}},"R"};
Rep3={{{1,0},{0}},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3};


HiggsDoublet={{{0,0},{1}},"C"};
RepScalar={HiggsDoublet};





RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;
Ysff=-GradYukawa[yt*YukawaDoublet[[1]]];


(* ::Section:: *)
(*SM quarks + gauge bosons*)


(* ::Subsection:: *)
(*SymmetryBreaking*)


vev={0,v,0,0};
SymmetryBreaking[vev]


(* ::Subsection:: *)
(*UserInput*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermoon
*)


(*
	Reps 1-4 are quarks,
	reps 5,6 are vector bosons
*)
(*left-handed top-quark*)
ReptL=CreateOutOfEq[{{1,1}},"F"];

(*right-handed top-quark*)
ReptR=CreateOutOfEq[{{2,1}},"F"];

(*light quarks*)
RepLight=CreateOutOfEq[{{1,2},3,4,5,6,7,8,9},"F"];

(*Vector bosons*)
RepGluon=CreateOutOfEq[{1},"V"];
RepW=CreateOutOfEq[{2},"V"];


ParticleList={ReptL,ReptR,RepW,RepGluon,RepLight};
(*
These particles do not have out-of-eq contributions
*)
(*LightParticles={3,4,5,6,7,8,9};*)
EqParticles={5};


(*Defining various masses and couplings*)


VectorMass=Join[
	Table[mg2,{i,1,RepGluon[[1]]//Length}],
	Table[mw2,{i,1,RepW[[1]]//Length}]];
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*
up to the user to make sure that the same order is given in the python code
*)
UserMasses={mq2,mq2,mw2,mg2}; 
UserCouplings=CouplingName;


(*
	output of matrix elements
*)
OutputFile="MatrixElements";
SetDirectory[DirectoryName[$InputFileName]];
RepOptional={c[1]->c[1]};
ParticleName={"TopL","TopR","W","Gluon"};
MatrixElements=ExportMatrixElements[OutputFile,ParticleList,EqParticles,UserMasses,UserCouplings,ParticleName,RepOptional];


MatrixElements//Expand


Import[OutputFile<>".hdf5"]


Import[OutputFile<>".hdf5","CouplingInfo"]


Import[OutputFile<>".hdf5","ParticleInfo"]


Import[OutputFile<>".hdf5","CouplingInfo"]


Import[OutputFile<>".hdf5","ParticleInfo"]