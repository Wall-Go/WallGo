(* ::Package:: *)

(*Quit[];*)


SetDirectory[NotebookDirectory[]];
$LoadGroupMath=True;
<<../DRalgo.m
<<matrixElements.m


(* ::Chapter:: *)
(*QCD+W boson*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2"};
RepAdjoint={{1,1},{2}};
RepScalar={};
CouplingName={gs,gw};


Rep1={{{1,0},{1}},"L"};
Rep2={{{1,0},{0}},"R"};
Rep3={{{1,0},{0}},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3};


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


(* ::Title:: *)
(*SM quarks + gauge bosons*)


(* ::Subtitle:: *)
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
ReptL=CreateOutOfEq[{1},"F"];

(*right-handed top-quark*)
ReptR=CreateOutOfEq[{2},"F"];

(*right-handed bottom-quark*)
RepbR=CreateOutOfEq[{3},"F"];

(*light quarks*)
RepLight=CreateOutOfEq[{4,5,6,7,8,9},"F"];

(*Vector bosons*)
RepGluon=CreateOutOfEq[{1},"V"];
RepW=CreateOutOfEq[{2},"V"];


ParticleList={ReptL,ReptR,RepbR,RepGluon,RepW,RepLight};
(*
These particles do not have out-of-eq contributions
*)
LightParticles={6};


(*Defining various masses and couplings*)


VectorMass=Join[Table[mg2,{i,1,RepGluon[[1]]//Length}],Table[mw2,{i,1,RepW[[1]]//Length}]];
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*
up to the user to make sure that the same order is given in the python code
*)
UserMasses={mq2,mw2,mg2}; 
UserCouplings={gs,gw};


SetDirectory[NotebookDirectory[]];
ParticleName={"TopL","TopR","BotR","Gluon","W"};
MatrixElements=ExportMatrixElements["MatrixElem",ParticleList,LightParticles,UserMasses,UserCouplings,ParticleName];


MatrixElements//Expand;


(*g g->g g*)
M[0,3,0,3]/.MatrixElements
(*g g->g g*)
M[3,3,3,3]/.MatrixElements
(*t g->t g*)
M[1,3,1,3]/.MatrixElements
(*t q->t q*)
5/4*M[1,5,1,5]/.MatrixElements


Import["MatrixElem.hdf5"]


Import["MatrixElem.hdf5","CouplingInfo"]


Import["MatrixElem.hdf5","ParticleInfo"]


Import["MatrixElem.hdf5","CouplingInfo"]


Import["MatrixElem.hdf5","ParticleInfo"]
