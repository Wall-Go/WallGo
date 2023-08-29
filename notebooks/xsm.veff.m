(* ::Package:: *)

Quit[];
(* Notebook to reproduce exactly Benoits result in xSM for p(T), serves as a template for Python implementation. By Tuomas Tenkanen, May 2023 in DESY.  *)


(* 
Different pieces;

DONE. Vtree;
DONE. MSbar parameters at tree-level;
DONE. T=0 relations for g, g' and gy;
DONE. mass eigenvalues;
DONE. T=0 1-loop master;
DONE. High T 1-loop masters;
DONE. Vcw;
DONE. VT;
DONE. Vacuum pressure and number of degrees of freedom;

DONE. Minimisation algorithm;
DONE. minima;

To be done later: routine to find Tc and pressure as function of T;


*)

(* tree-level Veff: *)
Vtree[h_,s_] :=((h^2 \[Mu]hsq)/2+(h^4 \[Lambda])/4+(\[Mu]ssq s^2)/2+(\[Lambda]s s^4)/4+1/4 h^2 \[Lambda]hs s^2);

(* tree-level MSbar stuff, i.e. require that at T=0 minimum is at v0 (and singlet vev at T=0 vanishes) and that we have two massive scalar eigenstates with masses Mh, Ms. These requirements allow us to solve tree-level relations for mass parameters and Higgs self interaction \[Lambda]. *)
e1=D[Vtree[h,s],{h,1}] /. s-> 0;
sol1=Solve[e1==0,h];
h0 = h /. sol1[[3]];
e2=mhsq==D[Vtree[h,s],{h,2}]/. s-> 0  /. h-> h0;
e3=mssq==D[Vtree[h,s],{s,2}] /. s-> 0 /. h-> v0;
sol2=Solve[e2,\[Mu]hsq];
\[Mu]hsq1 = \[Mu]hsq /. sol2[[1]];
sol3=Solve[e3,\[Mu]ssq] // Expand;
\[Mu]ssq1 = \[Mu]ssq /. sol3[[1]];

\[Mu]hsq1=-(mhsq/2);
\[Mu]ssq1=mssq-(v0^2 \[Lambda]hs)/2;

eq4=h0^2 == v0^2 /. \[Mu]hsq-> \[Mu]hsq1 // PowerExpand;
sol4=Solve[eq4,\[Lambda]];
\[Lambda]1 = \[Lambda] /. sol4[[1]];
\[Lambda]1=mhsq/(2 v0^2);


(* mass eigenvalues: *)
(* First gauge boson and top masses in terms of the background fields. I didn't bother to add derivation here, but it could be added. Instead, I just hardcode known relations: *)
mWsq=1/4 gsq h^2;
mZsq=1/4 (gsq+g1sq)h^2;
m\[Gamma]sq=0;

(* top mass eigenvalue: *)
mtsq =1/2 gysq h^2;


(* scalar mass eigenvalues: *)

(* parametrize the Higgs field in terms of background v0 and fluctuations \[Phi]i. Note that here v0 is just arbitrary background field and not the one at T=0, despite same notation. I just copy pasted this piece of code from my past codes! *)

phi1 = 1/Sqrt[2] ({
 {\[Phi]2+I \[Phi]1},
 {(v0+\[Phi]3+I \[Phi]4)}
});
phi1c = Transpose[1/Sqrt[2] ({
 {\[Phi]2-I \[Phi]1},
 {(v0+\[Phi]3-I \[Phi]4)}
})];

(* gauge fields: *)

(* Covariant derivative: (\[Beta] is used to take I->-I) *)
Dphi1 =1/Sqrt[2] ({
 {d\[Phi]2+\[Beta] I d\[Phi]1},
 {d\[Phi]3+\[Beta] I d\[Phi]4}
}) +x(-\[Beta] I g1 1/2 B0 1/Sqrt[2] ({
 {\[Phi]2+\[Beta] I \[Phi]1},
 {(v0+\[Phi]3+\[Beta] I \[Phi]4)}
})- \[Beta] I g/2 ({
 {W3, W1-\[Beta] I W2},
 {W1+ \[Beta] I W2, -W3}
}) . ({
 {\[Phi]2+\[Beta] I \[Phi]1},
 {(v0+\[Phi]3+\[Beta] I \[Phi]4)}
}) 1/Sqrt[2] );
Dphi1Dag = Transpose[Dphi1/. \[Beta]-> -\[Beta]];

(* This gauge fixing term is NOT needed here!*)
(* R\[Xi] gauge: *)
LagGF = 1/2 (1/\[Xi]B (dB0 + g1/2 \[Xi]B  x v0 (-\[Phi]4) )^2 + 1/\[Xi]A (dW1 -g/2 \[Xi]A x v0 \[Phi]1)^2 + 1/\[Xi]A (dW2 -g/2 \[Xi]A x v0 \[Phi]2)^2 + 1/\[Xi]A (dW3 - g/2 \[Xi]A x v0 (-\[Phi]4))^2) /. \[Xi]A-> \[Xi] /. \[Xi]B-> \[Xi];

(* Use notation similar to CORE: *)

(* NOTE, CAUTION: WRONG CONVENTION FOR Wpm: {{Wp->(-I W1+W2)/Sqrt[2],Wm->(I W1+W2)/Sqrt[2]}} *)
EigenFields = { W2->1/Sqrt[2] (Wp+Wm)  , W1-> I/Sqrt[2] (Wp-Wm) , W3 -> A g1/Sqrt[g^2+g1^2]+Z g/Sqrt[g^2+g1^2] , B0 -> -Z g1/Sqrt[g^2+g1^2]+A g/Sqrt[g^2+g1^2],dW2-> 1/Sqrt[2] (dWp+dWm)  , dW1-> I/Sqrt[2] (dWp-dWm) , dW3 -> dA g1/Sqrt[g^2+g1^2]+dZ g/Sqrt[g^2+g1^2] , dB0 -> -dZ g1/Sqrt[g^2+g1^2]+dA g/Sqrt[g^2+g1^2]};
subFields={\[CapitalSigma]1->\[CapitalSigma]m/Sqrt[2]+\[CapitalSigma]p/Sqrt[2],\[CapitalSigma]2->-((I \[CapitalSigma]m)/Sqrt[2])+(I \[CapitalSigma]p)/Sqrt[2],\[Phi]4-> z,\[Phi]1->Gm/Sqrt[2]+Gp/Sqrt[2],\[Phi]2->(I Gm)/Sqrt[2]-(I Gp)/Sqrt[2],\[Phi]3->h Cos[\[Theta]]+\[CapitalSigma]0 Sin[\[Theta]],s->\[CapitalSigma]0 Cos[\[Theta]]-h Sin[\[Theta]]};

subFieldsKinetic={d\[CapitalSigma]1->d\[CapitalSigma]m/Sqrt[2]+d\[CapitalSigma]p/Sqrt[2],d\[CapitalSigma]2->-((I d\[CapitalSigma]m)/Sqrt[2])+(I d\[CapitalSigma]p)/Sqrt[2],d\[Phi]4-> dz,d\[Phi]1->dGm/Sqrt[2]+dGp/Sqrt[2],d\[Phi]2->(I dGm)/Sqrt[2]-(I dGp)/Sqrt[2],d\[Phi]3->dh Cos[\[Theta]]+d\[CapitalSigma]0 Sin[\[Theta]],d\[CapitalSigma]3->d\[CapitalSigma]0 Cos[\[Theta]]-dh Sin[\[Theta]]}; (* these kinetic terms substitutions are not needed below, since we just need to find the mass matrix and its eigenvalues, and we dont go beyond one-loop! *)

(*(* note: L = T + V, potential with plus sign!*)*)


V1gauge = kinetic x^2  (Dphi1Dag . Dphi1 ) +  0 x^2 GF LagGF /. \[Beta]-> 1;

V1scalar = x^2 \[Mu]hsq phi1c . phi1 + x^4 \[Lambda] (phi1c . phi1)^2 + x^2 1/2 \[Mu]ssq S^2 +x^4 1/4 \[Lambda]s S^4 + x^4 1/2 \[Lambda]hs S^2 (phi1c . phi1)  /. S-> s + \[Rho];

V1 = V1scalar + gauge V1gauge;

V2 = V1[[1]][[1]] /. v0-> v0/x  /. \[Rho]-> \[Rho]/x// Expand;
V3 = Series[V2,{x,0,2}];

(*Print["Tree-level Veff:"]*)
Vtree1 = SeriesCoefficient[V3,0] // Expand;
kk;
V4 = SeriesCoefficient[V3,2];

Vbilin = V4;  

(* The goal, the scalar mass matrix:  *)
Mmatrix = 2({
 {Coefficient[Vbilin,\[Phi]1^2] , 0, 0, 0, 0},
 {0, Coefficient[Vbilin,\[Phi]2^2] , 0, 0, 0},
 {0, 0, Coefficient[Vbilin,\[Phi]3^2] , 0, 1/2 Coefficient[Vbilin,\[Phi]3 s]},
 {0, 0, 0, Coefficient[Vbilin,\[Phi]4^2] , 0},
 {0, 0, 1/2 Coefficient[Vbilin,\[Phi]3 s], 0, Coefficient[Vbilin,s^2] }
});
temp2= 1/2 (({
 {\[Phi]1, \[Phi]2, \[Phi]3, \[Phi]4, s}
}) . Mmatrix . ({
 {\[Phi]1},
 {\[Phi]2},
 {\[Phi]3},
 {\[Phi]4},
 {s}
}));

temp22= temp2[[1]][[1]];

Vbilin-temp22 /. gauge-> 1 /. kinetic -> 0 /. GF-> 0// Expand;



(*Print["Mass (squared) eigenvalues: "]*)
eig=Eigenvalues[Mmatrix] /. v0-> h /. \[Rho]-> s // FullSimplify

(* Goldstone mass eigenvalue, this is triple degenerate: *)
mGsq=eig[[1]];
eig[[2]];
eig[[3]];
(* Higgs and singlet mass eigenvalues: *)
mminussq=eig[[4]];
mplussq=eig[[5]];


(* Coleman Weinberg T=0 Veff:*)

(* In D-dimensions: *)
Jcw[msq]:= -(1/2) ((\[Mu]sq Exp[EulerGamma])/(4\[Pi]))^\[Epsilon] (msq)^(D/2)/(4\[Pi])^(D/2) Gamma[-(D/2)] /. D-> 4-2\[Epsilon];
(* Find D=4-2\[Epsilon] expressions for scalar/top and gauge field cases separately: *)
(msq^2/(4\[Pi])^2 1/4)^-1 SeriesCoefficient[Series[Jcw[msq],{\[Epsilon],0,0}],0] // Expand;
(msq^2/(4\[Pi])^2 3/4)^-1 SeriesCoefficient[Series[(D-1)Jcw[msq] /. D-> 4-2\[Epsilon],{\[Epsilon],0,0}],0] // Expand;

(* Note that here I cheat: I simply neglect 1/\[Epsilon] pole. Properly it would be removed by the counterterms, but I dont bother to include them here! *)
Jcwscalartop[msq_,\[Mu]sq_]:= (msq^2/(4\[Pi])^2 1/4)(-(3/2)+Log[msq/\[Mu]sq]);
Jcwgauge[msq_,\[Mu]sq_]:= (msq^2/(4\[Pi])^2 3/4)(-(5/6)+Log[msq/\[Mu]sq]);

(* combine all pieces: (note that fermionic contribution gets (-1 due to fermion loop, factor 4 from Dirac trace, and top contribution is Nc-degenerate due to SU(Nc) colour!))*)
Vcw = scalar(3 Jcwscalartop[mGsq,\[Mu]sq] + Jcwscalartop[mminussq,\[Mu]sq] + Jcwscalartop[mplussq,\[Mu]sq]) + top  ((-4Nc)Jcwscalartop[mtsq,\[Mu]sq])+ gauge( 2Jcwgauge[mWsq,\[Mu]sq] + Jcwgauge[mZsq,\[Mu]sq]);

(* 1-loop thermal masters, without high T expansion, and WITHOUT any resummation: *)
(* y = m/T *)
Jb[y_?NumericQ]:=((4\[Pi])/(2\[Pi])^3) NIntegrate[x^2 Log[1-Exp[-Sqrt[x^2+y^2]]],{x,0,Infinity}]; 
Jf[y_?NumericQ]:=((4\[Pi])/(2\[Pi])^3) NIntegrate[x^2 Log[1+Exp[-Sqrt[x^2+y^2]]],{x,0,Infinity}]; 

(* finite T part: *)
(* Note that I add overall T^4 here, since I didnt add it to thermal functions. Also note that I add all massless degrees of freedom, including ghosts corresponding to all gauge fields. In total, this amounts correct vacuum pressure p0 = gstar(\[Pi]^2/90T^4), where gstar = 107.75 = gstarSM + 1 at massless limit (i.e. when background fields go to zero). *)
VT =T^4 (scalar(3 Jb[Sqrt[mGsq]/T] + Jb[Sqrt[mminussq]/T] + Jb[Sqrt[mplussq]/T]) +  top ((-4Nc)Jf[Sqrt[mtsq]/T])+ gauge (4-1)( 2 Jb[Sqrt[mWsq]/T] + Jb[Sqrt[mZsq]/T])
+ masslessDoF((1+(Nc^2-1))(4-1)Jb[0/T]+(-1) (90-12)Jf[0/T]+ghost(-1)(((2Jb[0/T] + Jb[0/T]) + Jb[0/T] + (Nc^2-1)Jb[0/T]))));

Veff =    tree Vtree[h,s] +  cw Vcw + vt VT;
(*Veff=Vtree[h,s]+Vcw+VT;*)


(* Test minimisation:*)

NN = 1;
Nf = 3;
Nc = 3;

{Ms,\[Lambda]hs,\[Lambda]s} = {160,1.6,1.0}; (* BM point by Tuomas and Jorinde: this gets stuck to singlet phase in Benoits low order approximation without daisies etc. *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {160,1.2,1.0}; (* new BM point from Benoits paper!*)*)
{Ms,\[Lambda]hs,\[Lambda]s} = {103.79,0.7152,1.0}; (* new BM point from Benoits paper!*)

{Mh,Mw,Mz, Mt,v0} = {125,80,91,173,246.22}; (* rougher number by Benoit *)

Gf = 1.1663787 10^-5; (* this is not used by Benoit! *)
(*Mt = 172.76;
Mw = 80.379;
Mz = 91.1876;
gSsq = 1.48409;*)

g0sq = 4Sqrt[2]Gf Mw^2; (* I normally use this! *)

(* but Benoit uses this: *)
g0sq=((2 Mw)/v0)^2 (* match to Benoit *);

gsq = g0sq;
g1sq = g0sq (Mz^2/Mw^2-1);
gysq = 1/2 g0sq Mt^2/Mw^2;
vev = Sqrt[((4 Mw^2)/g0sq)];

\[Mu]sqBenoit = Mh^2; 

\[Mu]sq=\[Mu]sqBenoit; (* I normally use \[Mu] = A \[Pi] T, where A is around 0.5 to 2*)

(* Requirement of T=0 minimum at v0 and eigenstates at v0 with masses Mh and Ms allow us to solve: *)
\[Mu]hsq=-(Mh^2/2);
\[Mu]ssq=Ms^2-(v0^2 \[Lambda]hs)/2;

\[Lambda]=Mh^2/(2 v0^2);

(* this temperature range was used by Tuomas and Jorinde for N3LO pressure, but we dont yet use this here! *)
{Tmin,Tmax,dT}= {50,130,0.1};


(***********)

T =   100;

(* A substitution to remove debug flags: *)
removeFlags = {masslessDoF-> 1, otherFermions-> 1, gluons-> 1, photon-> 1, gauge-> 1 , scalar -> 1,top-> 1,ghost-> 1};

Print["Compare value of the Veff with Benoit at some arbitrary point: "]
Veff /. removeFlags /. h-> 110 /. s-> 130 // Expand // N  // Quiet
%/.{tree->1,cw->1,vt->1}
(*%/.{top->1,gauge->0,scalar->0}*)


(53574.70475221^2 1)(-(3/2)+Log[53574.70475221/\[Mu]sq])


(***********)
(* Minimisation: *)
(***********)

acc = 10; (* this controls the number of search points below in DifferentialEvolution method for NMinimize. In practise this has to be large number, around 100 or even 1000, but that makes the code awfully slow so here I use very rough accuracy. Minima to be found cannot be expected to be to accurate, but even acc == 10 finds minima that approximately match those found by Benoit.   *)

scalingFactor=0.6; (** Mathematica default is 0.6 **)

(* for a quick test, fix T here: *)




veff1 = Veff /. removeFlags;

Print["This finds the GLOBAL minimum of the potential: "]
min1=NMinimize[Re[veff1],{h,s},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet


(* ::Input:: *)
(**)
