(* ::Package:: *)

(*Quit[];*)


SetDirectory[DirectoryName[$InputFileName]];
(*Put this if you want to create multiple model-files with the same kernel*)
$GroupMathMultipleModels=True;
$LoadGroupMath=True;
<<DRalgo`;
<<matrixElements`;


(* ::Chapter:: *)
(*QCD*)


(* ::Section::Closed:: *)
(*Model*)


Group={"SU3"};
RepAdjoint={{1,1}};
RepScalar={};
CouplingName={gs};


Rep1={{{1,0}},"L"};
Rep2={{{1,0}},"R"};
RepFermion1Gen={Rep1,Rep2};


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen,RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


(* ::Title:: *)
(*A model with 6 quarks and 1 gluon*)


(* ::Subtitle:: *)
(*UserInput*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermoon
*)


(*Below
rep 1-6 are quarks,
rep 7 is a gluon
*)
Rep1=CreateOutOfEq[{1,2},"F"];
Rep2=CreateOutOfEq[{3,4},"F"];
Rep3=CreateOutOfEq[{5,6},"F"];
Rep4=CreateOutOfEq[{7,8},"F"];
Rep5=CreateOutOfEq[{9,10},"F"];
Rep6=CreateOutOfEq[{11,12},"F"];
RepGluon=CreateOutOfEq[{1},"V"];
(*check*)
(*Rep2=CreateOutOfEq[{3,4,...,12},"F"];*)


ParticleList={Rep1,RepGluon,Rep2,Rep3,Rep4,Rep5,Rep6};
(*
These particles do not have out-of-eq contributions
*)
LightParticles={3,4,5,6,7};


(*Defining various masses and couplings*)


VectorMass=Table[mg2,{i,1,Length[gvff]}];
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*
up to the user to make sure that the same order is given in the python code
*)
UserMasses={mq2,mg2};
UserCouplings={gs};


SetDirectory[DirectoryName[$InputFileName]];
OutputFile="../MatrixElements";
ParticleName={"Top","Gluon"};
MatrixElements=ExportMatrixElements[OutputFile,ParticleList,LightParticles,UserMasses,UserCouplings,ParticleName];
