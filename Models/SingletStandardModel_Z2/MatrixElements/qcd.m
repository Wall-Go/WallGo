(* ::Package:: *)

(*Quit[];*)


(*SetDirectory[NotebookDirectory[]];*)
(*Put this if you want to create multiple model-files with the same kernel*)
$GroupMathMultipleModels=True;
$LoadGroupMath=True;
<<WallGoMatrix`


(* ::Chapter:: *)
(*QCD*)


(* ::Section:: *)
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


ImportModel[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];


(* ::Section:: *)
(*A model with 6 quarks and 1 gluon*)


(* ::Subsection:: *)
(*UserInput*)


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermion
*)


(*Below
rep 1-6 are quarks,
rep 7 is a gluon
*)
Rep1=CreateParticle[{1,2},"F"];
RepGluon=CreateParticle[{1},"V"];


(*Defining various masses and couplings*)


VectorMass=Table[mg2,{i,1,Length[gvff]}];
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
ScalarMass={};
ParticleMasses={VectorMass,FermionMass,ScalarMass};
(*
up to the user to make sure that the same order is given in the python code
*)
UserMasses={mq2,mg2};
UserCouplings={gs};


(*
These particles do not necessarily have to be out of equilibrium
the remainin particle content is set as light
*)
ParticleList={Rep1,RepGluon};
ParticleName={"Top", "Gluon"};


Print[$ScriptCommandLine]


(*
	output of matrix elements
*)
(*SetDirectory[NotebookDirectory[]];*)
OutputFile="output/matrixElements.qcd";
MatrixElements=ExportMatrixElements[
	OutputFile,
	ParticleList,
	UserMasses,
	UserCouplings,
	ParticleName,
	ParticleMasses,
	{TruncateAtLeadingLog->True,Format->{"json","txt"}}];



