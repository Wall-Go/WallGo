(* ::Package:: *)

(*Quit[];*)


SetDirectory[DirectoryName[$InputFileName]];
(*Put this if you want to create multiple model-files with the same kernel*)
$GroupMathMultipleModels=True;
$LoadGroupMath=True;
<<DRalgo`;
<<matrixElements`;


(* ::Chapter:: *)
(*SM+sr1*)


(*see 2102.11145 [hep-ph]*)


(* ::Section::Closed:: *)
(*Model*)


HypY={Yl,Ye,Yq,Yu,Yd,Y\[Phi],Y\[Eta]};
repY=Thread[HypY->{-1,-2,1/3,4/3,-(2/3),1,2}];


Group={"SU3","SU2","U1"};
RepAdjoint={{1,1},{2},0};
scalar1={{{0,0},{1},Y\[Phi]/2},"C"};
scalar2={{{0,0},{0},0},"R"};
RepScalar={scalar1,scalar2}/.repY;
CouplingName={g3,gw,g1};


Rep1={{{1,0},{1},Yq/2},"L"};
Rep2={{{1,0},{0},Yu/2},"R"};
Rep3={{{1,0},{0},Yd/2},"R"};
Rep4={{{0,0},{1},Yl/2},"L"};
Rep5={{{0,0},{0},Ye/2},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5}/.repY;


(* ::Text:: *)
(*The input for the gauge interactions to DRalgo are then given by*)


RepFermion1Gen={Rep1,Rep2,Rep3,Rep1,Rep2,Rep3,Rep1,Rep2,Rep3,Rep4,Rep5}/.repY;
RepFermion3Gen={RepFermion1Gen}//Flatten[#,1]&;
(*RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;*)


(* ::Text:: *)
(*The first element is the vector self-interaction matrix:*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar]/.repY;


InputInv={{1,1},{True,False}};
MassTerm1=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,2},{True,True}};
MassTerm2=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;


VMass=(
	+m1*MassTerm1
	+\[Mu]\[Sigma]/2*MassTerm2
	);


\[Mu]ij=GradMass[VMass[[1]]]//Simplify//SparseArray;


QuarticTerm1=MassTerm1[[1]]^2;
QuarticTerm2=MassTerm2[[1]]^2;
QuarticTerm3=MassTerm1[[1]]*MassTerm2[[1]];


VQuartic=(
	+\[Lambda]1H*QuarticTerm1
	+\[Lambda]\[Sigma]/4*QuarticTerm2
	+\[Lambda]m/2*QuarticTerm3
	);


\[Lambda]4=GradQuartic[VQuartic];


InputInv={{1,1,2},{True,False,True}};
CubicTerm1=CreateInvariant[Group,RepScalar,InputInv][[1]]//Simplify;
InputInv={{2,2,2},{True,True,True}};
CubicTerm2=CreateInvariant[Group,RepScalar,InputInv][[1]]//Simplify;


VCubic=(
	+\[Mu]m/2*CubicTerm1
	+\[Mu]3/3*CubicTerm2
	);


\[Lambda]3=GradCubic[VCubic];


InputInv={{2},{True}};
TadpoleTerm1=CreateInvariant[Group,RepScalar,InputInv][[1]]//Simplify;


VTadpole=\[Mu]1*TadpoleTerm1;


\[Lambda]1=GradTadpole[VTadpole];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet1=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;


Ysff=-yt1*GradYukawa[YukawaDoublet1[[1]]];


YsffC=SparseArray[Simplify[Conjugate[Ysff]//Normal,Assumptions->{yt1>0}]];


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


vev={0,v,0,0,0};
SymmetryBreaking[vev]


(*left-handed top-quark*)
ReptL=CreateOutOfEq[{{1,1}},"F"];

(*right-handed top-quark*)
ReptR=CreateOutOfEq[{2},"F"];

(*left-handed bottom-quark*)
RepbL=CreateOutOfEq[{{1,2}},"F"];

(*light quarks*)
RepLight=CreateOutOfEq[Range[3,2*4+3],"F"];

(*Vector bosons*)
RepGluon=CreateOutOfEq[{1},"V"];
RepW=CreateOutOfEq[{{2,1}},"V"];
RepZ=CreateOutOfEq[{{3,1}},"V"];


ParticleList={ReptL,ReptR,RepGluon,RepW,RepZ,RepbL,RepLight};
(*
These particles do not have out-of-eq contributions
*)
LightParticles=Range[6,Length[ParticleList]];


VectorMass=Join[
	Table[mg2,{i,1,RepGluon[[1]]//Length}],
	Table[mw2,{i,1,RepW[[1]]//Length}],
	Table[mz2,{i,1,RepZ[[1]]//Length}]];
FermionMass=Table[mq2,{i,1,Length[gvff[[1]]]}];
(*
up to the user to make sure that the same order is given in the python code
*)
UserMasses={mq2,mq2,mg2,mw2,mz2}; 
UserCouplings={g3,gw,g1};


SetDirectory[DirectoryName[$InputFileName]];
OutputFile="MatrixElements";
(*RepOptional={c[1]->0,c[2]->0};*)
RepOptional={};
ParticleName={"TopL","TopR","Gluon","W","Z"};
MatrixElements=ExportMatrixElements[OutputFile,ParticleList,LightParticles,UserMasses,UserCouplings,ParticleName,RepOptional];


(*tL q->tL q*)
M[0,3,0,3]/.MatrixElements;
%/.{c[1]->0,c[2]->0}//FullSimplify
(*tR q->tR q*)
M[1,3,1,3]/.MatrixElements;
%/.{c[1]->0,c[2]->0}//FullSimplify
(*tL tL->gg*)
M[0,0,2,2]/.MatrixElements;
%/.{c[1]->0,c[2]->0}
(*tR tR->gg*)
M[1,1,2,2]/.MatrixElements;
%/.{c[1]->0,c[2]->0}
(*tL g->tL g*)
M[0,2,0,2]/.MatrixElements//FullSimplify;
%/.{c[1]->0,c[2]->0}
(*tR g->tR g*)
M[1,2,1,2]/.MatrixElements//FullSimplify;
%/.{c[1]->0,c[2]->0}


(*off-diagonal entries do not agree*)
M[1,0,0,1]/.MatrixElements//FullSimplify;
%/.{c[1]->0,c[2]->0}
M[0,1,0,1]/.MatrixElements//FullSimplify;
%/.{c[1]->0,c[2]->0}
(*diagonal entries agree*)
M[1,1,0,0]/.MatrixElements//FullSimplify;
%/.{c[1]->0,c[2]->0}
M[0,0,1,1]/.MatrixElements//FullSimplify;
%/.{c[1]->0,c[2]->0}
M[0,0,0,0]/.MatrixElements;
%/.{c[1]->0,c[2]->0}//Simplify
M[1,1,1,1]/.MatrixElements;
%/.{c[1]->0,c[2]->0}//Simplify


(*result pure qcd*)
1/12 (-((32 s^2)/(3 t u))+(32 (s^2+u^2))/(t-msq[1])^2+(32 (s^2+t^2))/(u-msq[1])^2)//Simplify


(*crosscheck for Matrix for collisions*)
M[1,1,1,1]+M[1,0,0,1]+M[1,1,0,0];
%/.MatrixElements/.{c[0]->1};
%/.{c[1]->0,c[2]->0}//FullSimplify
M[0,0,0,0]+M[0,1,0,1]+M[0,0,1,1];
%/.MatrixElements/.{c[0]->1};
%/.{c[1]->0,c[2]->0}//FullSimplify


(*check eigenvalues*)
{{M[0,0,0,0],M[0,1,0,1]},{M[1,0,0,1],M[1,1,1,1]}};
%/.MatrixElements/.{c[0]->1};
%/.{c[1]->0,c[2]->0};
Eigenvalues[%]//FullSimplify(*//Total//Simplify*)


(*sum over all tL,tR channels*)
M[1,1,1,1]+M[0,0,0,0]+M[1,0,0,1]+M[0,1,0,1]+M[1,1,0,0]+M[0,0,1,1];
%/.MatrixElements/.{c[0]->1};
%/.{c[1]->0,c[2]->0}//FullSimplify


(*pure qcd result differs by factor 2 which is compensated by additonal factor 2 in EOM*)
1/6 (-((16 s^2)/(3 t u))-(32 u^2)/(3 s t)+(16 (t^2+u^2))/s^2+(16 (s^2+u^2))/(t-msq[1])^2+(8 (s^2+t^2))/(u-msq[1])^2)//FullSimplify
