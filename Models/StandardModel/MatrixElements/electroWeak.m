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
(*QCD+W boson*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2"};
RepAdjoint={{1,1},{2}};
CouplingName={gs,gw};


Rep1={{{1,0},{1}},"L"};
Rep2={{{1,0},{0}},"R"};
Rep3={{{1,0},{0}},"R"};
Rep4={{{0,0},{1}},"L"};
Rep5={{{0,0},{0}},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5};



HiggsDoublet={{{0,0},{1}},"C"};
RepScalar={HiggsDoublet};


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;
Ysff=-GradYukawa[yt*YukawaDoublet[[1]]];


ImportModel[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];


(* ::Section:: *)
(*SM quarks + gauge bosons + leptons*)


(* ::Subsection:: *)
(*SymmetryBreaking*)


vev={0,v,0,0};
SymmetryBreaking[vev]


(* ::Subsection:: *)
(*UserInput*)


PrintFieldRepPositions["Fermion"]


(*
In DRalgo fermions are Weyl.
So to create one Dirac we need
one left-handed and
one right-handed fermoon
*)


(*left-handed top-quark*)
ReptL=CreateParticle[{{1,1}},"F", mq2, "TopL"];

(*right-handed top-quark*)
ReptR=CreateParticle[{{2,1}},"F", mq2, "TopR"];

(*light quarks*)
RepLightQ = CreateParticle[{{1,2},3,6,7,8,11,12,13},"F",mq2, "LightQuark"];

(*left-handed leptons*)
RepLepL = CreateParticle[{4,9,14},"F", ml2,"LepL"];

(*right-handed leptons -- these don't contribute*)
RepLepR = CreateParticle[{5,10,15},"F",mlr2,"LepR"];

(*Vector bosons*)
RepGluon=CreateParticle[{1},"V",mg2,"Gluon"];

(*We are approximating the W and the Z as the same particle*)
RepW=CreateParticle[{{2,1}},"V",mw2,"W"];

(*Higgs*)
RepH = CreateParticle[{1},"S",ms2,"H"];


(*
These particles do not necessarily have to be out of equilibrium
*)
ParticleList={ReptL,ReptR,RepLightQ,RepGluon,RepW};
(*
Light particles are never incoming particles 
*)
LightParticleList={RepLepL,RepLepR, RepH};


(*
	output of matrix elements
*)
OutputFile="matrixElements.ew";
SetDirectory[NotebookDirectory[]];
MatrixElements=ExportMatrixElements[
	OutputFile,
	ParticleList,
	LightParticleList,
	{TruncateAtLeadingLog->True,Replacements->{yt->0},Format->{"json","txt"}}];
