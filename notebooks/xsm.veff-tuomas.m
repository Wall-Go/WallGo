(* ::Package:: *)

Quit[];
(* Notebook to reproduce exactly Benoits result in xSM for p(T), serves as a template for Python implementation. By Tuomas Tenkanen, May 2023 in DESY.  *)

(* Modified further by Tuomas, September 2023 in Helsinki and Bern: compared vanilla 4d method with Benoit, and added LO 3d EFT results  *)


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
DONE: routine to find Tc and pressure as function of T;

DONE: 3d Veff with LO DR, direct minimisation and strict expansion (at LO these are indentical, but other uses analytic expressions for minima)


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

(* note that Jb[0/T] == -(\[Pi]^2/90) == -0.10966227106381857` *)

Veff =    tree Vtree[h,s] +  cw Vcw + vt VT;
(*Veff=Vtree[h,s]+Vcw+VT;*)


removeFlags = {masslessDoF-> 1, otherFermions-> 1, gluons-> 1, photon-> 1, gauge-> 1 , scalar -> 1,top-> 1,ghost-> 1};
removeFlags2 = {tree->1,cw->1,vt->1};
Veff /. removeFlags /. removeFlags2



(* function for Veff in 4d approach *)
Veff4d[T_,\[Mu]sq_,Ms_,\[Lambda]hs_,\[Lambda]s_,h_,s_]:=Module[{veff,NN,Nf,Nc,Mh,Mw,Mz, Mt,v0,Gf,g0sq,gsq,g1sq,gysq,vev,\[Mu]hsq,\[Mu]ssq,\[Lambda]},

NN = 1;
Nf = 3;
Nc = 3;

{Mh,Mw,Mz, Mt,v0} = {125,80,91,173,246.22}; (* rougher number by Benoit *)

Gf = 1.1663787 10^-5; (* this is not used by Benoit! *)
(*Mt = 172.76;
Mw = 80.379;
Mz = 91.1876;
gSsq = 1.48409;*)

(*g0sq = 4Sqrt[2]Gf Mw^2;*) (* I normally use this! *)

(* but Benoit uses this: *)
g0sq=((2 Mw)/v0)^2 (* match to Benoit *);

gsq = g0sq;
g1sq = g0sq (Mz^2/Mw^2-1);
gysq = 1/2 g0sq Mt^2/Mw^2;
vev = Sqrt[((4 Mw^2)/g0sq)];

(*\[Mu]sqBenoit = Mh^2; *)
(*\[Mu]sq=\[Mu]sqBenoit;*) (* I normally use \[Mu] = A \[Pi] T, where A is around 0.5 to 2*)

(* Requirement of T=0 minimum at v0 and eigenstates at v0 with masses Mh and Ms allow us to solve: *)
\[Mu]hsq=-(Mh^2/2);
\[Mu]ssq=Ms^2-(v0^2 \[Lambda]hs)/2;
\[Lambda]=Mh^2/(2 v0^2);

(* 1-loop thermal masters, without high T expansion, and WITHOUT any resummation: *)
(* y = m/T *)
Jb[y_?NumericQ]:=((4\[Pi])/(2\[Pi])^3) NIntegrate[x^2 Log[1-Exp[-Sqrt[x^2+y^2]]],{x,0,Infinity}]; 
Jf[y_?NumericQ]:=((4\[Pi])/(2\[Pi])^3) NIntegrate[x^2 Log[1+Exp[-Sqrt[x^2+y^2]]],{x,0,Infinity}]; 

(* note that since using thermal functions Jb and Jf just like that (performing NIntegrate over and over again at each call) makes findTc4d below super slow *)

(* different pieces of this were computed separately above: *)
veff = (h^4 \[Lambda])/4+1/4 h^2 s^2 \[Lambda]hs+(s^4 \[Lambda]s)/4+(h^2 \[Mu]hsq)/2+(s^2 \[Mu]ssq)/2+T^4 (-7.045800919905512`- 0.32898681319145573` Nc^2+0.10966227106381857` (-1+Nc^2)+3 (2 Jb[Sqrt[gsq h^2]/(2 T)]+Jb[Sqrt[(g1sq+gsq) h^2]/(2 T)])+3 Jb[Sqrt[h^2 \[Lambda]+(s^2 \[Lambda]hs)/2+\[Mu]hsq]/T]+Jb[1/(2 T) (\[Sqrt](h^2 (6 \[Lambda]+\[Lambda]hs)+s^2 (\[Lambda]hs+6 \[Lambda]s)-Sqrt[h^4 (-6 \[Lambda]+\[Lambda]hs)^2+(s^2 (\[Lambda]hs-6 \[Lambda]s)+2 (\[Mu]hsq-\[Mu]ssq))^2+2 h^2 (s^2 (6 \[Lambda] (\[Lambda]hs-6 \[Lambda]s)+\[Lambda]hs (7 \[Lambda]hs+6 \[Lambda]s))+2 (6 \[Lambda]-\[Lambda]hs) (\[Mu]hsq-\[Mu]ssq))]+2 (\[Mu]hsq+\[Mu]ssq)))]+Jb[1/(2 T) (\[Sqrt](h^2 (6 \[Lambda]+\[Lambda]hs)+s^2 (\[Lambda]hs+6 \[Lambda]s)+Sqrt[h^4 (-6 \[Lambda]+\[Lambda]hs)^2+(s^2 (\[Lambda]hs-6 \[Lambda]s)+2 (\[Mu]hsq-\[Mu]ssq))^2+2 h^2 (s^2 (6 \[Lambda] (\[Lambda]hs-6 \[Lambda]s)+\[Lambda]hs (7 \[Lambda]hs+6 \[Lambda]s))+2 (6 \[Lambda]-\[Lambda]hs) (\[Mu]hsq-\[Mu]ssq))]+2 (\[Mu]hsq+\[Mu]ssq)))]-4 Nc Jf[Sqrt[gysq h^2]/(Sqrt[2] T)])+(3 gsq^2 h^4 (-(5/6)+Log[(gsq h^2)/(4 \[Mu]sq)]))/(512 \[Pi]^2)+(3 (g1sq+gsq)^2 h^4 (-(5/6)+Log[((g1sq+gsq) h^2)/(4 \[Mu]sq)]))/(1024 \[Pi]^2)-(gysq^2 h^4 Nc (-(3/2)+Log[(gysq h^2)/(2 \[Mu]sq)]))/(64 \[Pi]^2)+(3 (h^2 \[Lambda]+(s^2 \[Lambda]hs)/2+\[Mu]hsq)^2 (-(3/2)+Log[(h^2 \[Lambda]+(s^2 \[Lambda]hs)/2+\[Mu]hsq)/\[Mu]sq]))/(64 \[Pi]^2)+1/(1024 \[Pi]^2) (h^2 (6 \[Lambda]+\[Lambda]hs)+s^2 (\[Lambda]hs+6 \[Lambda]s)-Sqrt[h^4 (-6 \[Lambda]+\[Lambda]hs)^2+(s^2 (\[Lambda]hs-6 \[Lambda]s)+2 (\[Mu]hsq-\[Mu]ssq))^2+2 h^2 (s^2 (6 \[Lambda] (\[Lambda]hs-6 \[Lambda]s)+\[Lambda]hs (7 \[Lambda]hs+6 \[Lambda]s))+2 (6 \[Lambda]-\[Lambda]hs) (\[Mu]hsq-\[Mu]ssq))]+2 (\[Mu]hsq+\[Mu]ssq))^2 (-(3/2)+Log[1/(4 \[Mu]sq) (h^2 (6 \[Lambda]+\[Lambda]hs)+s^2 (\[Lambda]hs+6 \[Lambda]s)-Sqrt[h^4 (-6 \[Lambda]+\[Lambda]hs)^2+(s^2 (\[Lambda]hs-6 \[Lambda]s)+2 (\[Mu]hsq-\[Mu]ssq))^2+2 h^2 (s^2 (6 \[Lambda] (\[Lambda]hs-6 \[Lambda]s)+\[Lambda]hs (7 \[Lambda]hs+6 \[Lambda]s))+2 (6 \[Lambda]-\[Lambda]hs) (\[Mu]hsq-\[Mu]ssq))]+2 (\[Mu]hsq+\[Mu]ssq))])+1/(1024 \[Pi]^2) (h^2 (6 \[Lambda]+\[Lambda]hs)+s^2 (\[Lambda]hs+6 \[Lambda]s)+Sqrt[h^4 (-6 \[Lambda]+\[Lambda]hs)^2+(s^2 (\[Lambda]hs-6 \[Lambda]s)+2 (\[Mu]hsq-\[Mu]ssq))^2+2 h^2 (s^2 (6 \[Lambda] (\[Lambda]hs-6 \[Lambda]s)+\[Lambda]hs (7 \[Lambda]hs+6 \[Lambda]s))+2 (6 \[Lambda]-\[Lambda]hs) (\[Mu]hsq-\[Mu]ssq))]+2 (\[Mu]hsq+\[Mu]ssq))^2 (-(3/2)+Log[1/(4 \[Mu]sq) (h^2 (6 \[Lambda]+\[Lambda]hs)+s^2 (\[Lambda]hs+6 \[Lambda]s)+Sqrt[h^4 (-6 \[Lambda]+\[Lambda]hs)^2+(s^2 (\[Lambda]hs-6 \[Lambda]s)+2 (\[Mu]hsq-\[Mu]ssq))^2+2 h^2 (s^2 (6 \[Lambda] (\[Lambda]hs-6 \[Lambda]s)+\[Lambda]hs (7 \[Lambda]hs+6 \[Lambda]s))+2 (6 \[Lambda]-\[Lambda]hs) (\[Mu]hsq-\[Mu]ssq))]+2 (\[Mu]hsq+\[Mu]ssq))]);

Return[veff];

];



(* function for Veff in 4d approach *)
Veff3d[T_,\[Mu]sq_,Ms_,\[Lambda]hs_,\[Lambda]s_,h_,s_]:=Module[{h0Strict,s0Strict,Vh,Vs,veff,veff3dtree,NN,Nf,Nc,Mh,Mw,Mz, Mt,v0,Gf,g0sq,gsq,g1sq,gysq,vev,\[Mu]hsq,\[Mu]ssq,\[Lambda],p0,\[Lambda]hs3,\[Lambda]s3,\[Lambda]3,\[Mu]hsq3,\[Mu]ssq3,h3,s3},

NN = 1;
Nf = 3;
Nc = 3;

{Mh,Mw,Mz, Mt,v0} = {125,80,91,173,246.22}; (* rougher number by Benoit *)

Gf = 1.1663787 10^-5; (* this is not used by Benoit! *)
(*Mt = 172.76;
Mw = 80.379;
Mz = 91.1876;
gSsq = 1.48409;*)

(*g0sq = 4Sqrt[2]Gf Mw^2;*) (* I normally use this! *)

(* but Benoit uses this: *)
g0sq=((2 Mw)/v0)^2 (* match to Benoit *);

gsq = g0sq;
g1sq = g0sq (Mz^2/Mw^2-1);
gysq = 1/2 g0sq Mt^2/Mw^2;
vev = Sqrt[((4 Mw^2)/g0sq)];

(*\[Mu]sqBenoit = Mh^2; *)
(*\[Mu]sq=\[Mu]sqBenoit;*) (* I normally use \[Mu] = A \[Pi] T, where A is around 0.5 to 2*)

(* Requirement of T=0 minimum at v0 and eigenstates at v0 with masses Mh and Ms allow us to solve: *)
\[Mu]hsq=-(Mh^2/2);
\[Mu]ssq=Ms^2-(v0^2 \[Lambda]hs)/2;
\[Lambda]=Mh^2/(2 v0^2);

(* LO DR: *)

\[Lambda]3 = T \[Lambda];
\[Lambda]s3 = T \[Lambda]s;
\[Lambda]hs3 = T \[Lambda]hs;

(* one-loop thermal masses: *)
\[Mu]hsq3 = \[Mu]hsq + T^2 1/12 (3/4 (3gsq+g1sq)+Nc gysq + 6 \[Lambda] + 1/2 \[Lambda]hs);
\[Mu]ssq3 = \[Mu]ssq + T^2 ( 1/6 \[Lambda]hs + 1/4 \[Lambda]s);

(* 3d fields in terms of 4d fields: *)
h3 = h/Sqrt[T];
s3 = s/Sqrt[T];
p0 = (106.75`+1) \[Pi]^2/90 T^4; (* coefficient of the unit operator, LO pressure *)

veff3dtree = (h3^4 \[Lambda]3)/4+1/4 h3^2 s3^2 \[Lambda]hs3+(s3^4 \[Lambda]s3)/4+(h3^2 \[Mu]hsq3)/2+(s3^2 \[Mu]ssq3)/2;

veff = -p0 + T veff3dtree; (* convert 3d Veff to 4d units and also include vacuum pressure part, as Benoit has it in his 4d Veff. *)

(* strict expansion analytically :*)

(* 

V0 = (h^4 \[Lambda])/4+1/4 h^2 s^2 \[Lambda]hs+(s^4 \[Lambda]s)/4+(h^2 \[Mu]hsq)/2+(s^2 \[Mu]ssq)/2;
d1=D[V0,{h,1}]/. s->0
Solve[d1==0,h]
d2=D[V0,{s,1}]/. h->0
Solve[d2==0,s]

h0=(\[ImaginaryI] Sqrt[\[Mu]hsq])/Sqrt[\[Lambda]];
s0 = (\[ImaginaryI] Sqrt[\[Mu]ssq])/Sqrt[\[Lambda]s];
Vh=V0 /. h-> h0 /. s-> 0
Vs=V0 /. s-> s0 /. h-> 0

*)

(* analytic minima of tree-level potential, converted to 4d units: *)
h0Strict=Sqrt[T]((I Sqrt[\[Mu]hsq3])/Sqrt[\[Lambda]3]);
s0Strict = Sqrt[T]((I Sqrt[\[Mu]ssq3])/Sqrt[\[Lambda]s3]);

(* analytic results for LO Veff at each phase: *)
Vh = -p0+T(-(\[Mu]hsq3^2/(4 \[Lambda]3)));
Vs = -p0+T(-(\[Mu]ssq3^2/(4 \[Lambda]s3)));  

(* stuff for eventual numerics: *)

(*

V0 = (h^4 \[Lambda])/4+1/4 h^2 s^2 \[Lambda]hs+(s^4 \[Lambda]s)/4+(h^2 \[Mu]hsq)/2+(s^2 \[Mu]ssq)/2;
d1=D[V0,{h,1}]
d2=D[V0,{s,1}]
eq1 = d1==0;
eq2 = d2==0;

Solve[{eq1,eq2},{h,s}]

*)

(* first argument is Veff in terms of arbitrary background fields for minimisation. *)
(* 2nd and 3rd arguments are values of Veff at each phase *)
(* 4th and 5th argument are values of background fields at each minima, converted to 4d units. *)
Return[{veff,Vh,Vs,h0Strict,s0Strict}];

];



(* ::Input:: *)
(**)


(* function for finding phases and Tc, 4d approach *)
findTc4d[Ms_,\[Lambda]hs_,\[Lambda]s_,Tmin_,Tmax_,dT_,plotting_]:=Module[{sol,Tguess,t,phInt,psInt,u1,u2,u3,p11,p22,r1,r2,p0,p1,p2,p3,hmin,smin,hminlocal,sminlocal,min1,min2,min3,lim,hlim,slim,veff1,veff2,veff3,h,s,h0,s0,Tc,acc,scalingFactor,hGlobalList,sGlobalList,hLocalList,sLocalList,pGlobalList,phLocalList,psLocalList,\[Mu]sqBenoit,Mh,\[Mu]sq},

(***********)
(* Minimisation, loop over T range: *)
(***********)

acc = 10; (* this controls the number of search points below in DifferentialEvolution method for NMinimize. In practise this has to be large number, around 100 or even 1000, but that makes the code awfully slow so here I use very rough accuracy. Minima to be found cannot be expected to be to accurate, but even acc == 10 finds minima that approximately match those found by Benoit.   *)
scalingFactor=0.6; (** Mathematica default is 0.6 **)

(* for a quick test, fix T here: *)

(* test 1: *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {103.79,0.7152,1.0};*) (* new BM point from Benoits paper!*) (* My current toy algorithm gives Tc = 117.779 while benoits result is 117.73*)

(* test 2: *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {90.0,0.8,1.0};*) (* new BM point from Benoits paper!*) (* Tc = 87.699, Benoit gets 87.79 *)

(*T = 100;*)

(*Tmin = 80;
Tmax = 200;
dT = 20;*)

(* test 1 range: *)
(*Tmin = 100;
Tmax = 130;
dT = 5;*)

(* test 2 range: *)
(*Tmin = 60;
Tmax = 100;
dT = 5;*)


hGlobalList={};
sGlobalList={};
hLocalList={};
sLocalList={};
pGlobalList={};
phLocalList={};
psLocalList={};

Do[

Mh=125; 
\[Mu]sqBenoit = Mh^2; 
\[Mu]sq=\[Mu]sqBenoit; (* I normally use \[Mu] = A \[Pi] T, where A is around 0.5 to 2*)

veff1 = Veff4d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h,s]; (* for global minimum *)

h0 = 0.001;  
s0 = 0.001;  
veff2 = Veff4d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h,s0]; (* for local h minimum *)
veff3 = Veff4d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h0,s]; (* for local s minimum *)

(*Print["This finds the GLOBAL minimum of the potential: "]*)
min1=NMinimize[Re[veff1],{h,s},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet;

(*

NOTE: above minimisation is super slow due to way I have implemented thermal functions. This needs to be optimised in python implementation obviously.

*)

(* this control parameter, which should be small, allows me to find local broken minima *)
lim = 0.5;
hlim = lim;
slim = lim;

(* use control parameter here so that local minima are found: *)
min2=NMinimize[{Re[veff2],h>hlim},{h},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet;
min3=NMinimize[{Re[veff3],s>slim},{s},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet;

hmin = h /. min1[[2]];
smin = s /. min1[[2]];

hminlocal = h /. min2[[2]];
sminlocal = s /. min3[[2]];

(* pressure = -Veff4d  *)
p1 = -min1[[1]];
p2 = -min2[[1]];
p3 = -min3[[1]];

If[plotting==1,Print[{T,Abs[hmin]/T,Abs[smin]/T}]];

hGlobalList = Join[hGlobalList,{{T,Abs[hmin]/T}}];
sGlobalList = Join[sGlobalList,{{T,Abs[smin]/T}}];

hLocalList = Join[hLocalList,{{T,Abs[hminlocal]/T}}];
sLocalList = Join[sLocalList,{{T,Abs[sminlocal]/T}}];

p0 = (106.75`+1) \[Pi]^2/90 T^4;

pGlobalList = Join[pGlobalList,{{T,p1/p0}}];
phLocalList = Join[phLocalList,{{T,p2/p0}}];
psLocalList = Join[psLocalList,{{T,p3/p0}}];

,{T,Tmax, Tmin,-dT}];

r1=ListLinePlot[hGlobalList,PlotStyle->Blue];
r2=ListLinePlot[sGlobalList,PlotStyle->Red];
p11=ListLinePlot[hLocalList,PlotStyle->Cyan,AxesLabel->{"T","v/T"}];
p22=ListLinePlot[sLocalList,PlotStyle->Magenta];
If[plotting==1,Print[Show[p11,p22,r1,r2]]];

u1=ListLinePlot[pGlobalList,PlotStyle->Black];
u2=ListLinePlot[phLocalList, PlotStyle->Blue,AxesLabel->{"T","p/p0"}];
u3=ListLinePlot[psLocalList,PlotStyle->Red];

If[plotting==1,Print[Show[u2,u3,u1]]];

If[plotting==1,Print[Show[u2,u3]]];

phInt=Interpolation[phLocalList];
psInt=Interpolation[psLocalList];

(* Tc is found from condition that pressure of two broken phases are equal: *)
Tguess = 100;
sol=FindRoot[phInt[t]-psInt[t],{t,Tguess}];
(*Print["Tc for transition to EW phase: "];*)
Tc = t /. sol;

Return[Tc];

];




(* function for finding phases and Tc, 4d approach *)
findTc3d[Ms_,\[Lambda]hs_,\[Lambda]s_,Tmin_,Tmax_,dT_,plotting_]:=Module[{Tc0,veff0,VhStrict,VsStrict,p33,h0Strict,s0Strict,hLocalListStrict,sLocalListStrict,phLocalListStrict,psLocalListStrict,sol,Tguess,t,phInt,psInt,u1,u2,u3,p11,p22,r1,r2,p0,p1,p2,p3,hmin,smin,hminlocal,sminlocal,min1,min2,min3,lim,hlim,slim,veff1,veff2,veff3,h,s,h0,s0,Tc,acc,scalingFactor,hGlobalList,sGlobalList,hLocalList,sLocalList,pGlobalList,phLocalList,psLocalList,\[Mu]sqBenoit,Mh,\[Mu]sq},

(***********)
(* Minimisation, loop over T range: *)
(***********)

acc = 10; (* this controls the number of search points below in DifferentialEvolution method for NMinimize. In practise this has to be large number, around 100 or even 1000, but that makes the code awfully slow so here I use very rough accuracy. Minima to be found cannot be expected to be to accurate, but even acc == 10 finds minima that approximately match those found by Benoit.   *)
scalingFactor=0.6; (** Mathematica default is 0.6 **)

(* for a quick test, fix T here: *)

(* test 1: *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {103.79,0.7152,1.0};*) (* new BM point from Benoits paper!*) (* My current toy algorithm gives Tc = 117.779 while benoits result is 117.73*)

(* test 2: *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {90.0,0.8,1.0};*) (* new BM point from Benoits paper!*) (* Tc = 87.699, Benoit gets 87.79 *)

(*T = 100;*)

(*Tmin = 80;
Tmax = 200;
dT = 20;*)

(* test 1 range: *)
(*Tmin = 100;
Tmax = 130;
dT = 5;*)

(* test 2 range: *)
(*Tmin = 60;
Tmax = 100;
dT = 5;*)


hGlobalList={};
sGlobalList={};
hLocalList={};
sLocalList={};
pGlobalList={};
phLocalList={};
psLocalList={};

hLocalListStrict={};
sLocalListStrict={};
phLocalListStrict={};
psLocalListStrict={};


Do[

Mh=125; 
\[Mu]sqBenoit = Mh^2; 
\[Mu]sq=\[Mu]sqBenoit; (* I normally use \[Mu] = A \[Pi] T, where A is around 0.5 to 2*)

veff1 = Veff3d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h,s][[1]]; (* for global minimum *)

h0 = 0.001;  
s0 = 0.001;  
veff2 = Veff3d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h,s0][[1]]; (* for local h minimum *)
veff3 = Veff3d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h0,s][[1]]; (* for local s minimum *)

(*Print["This finds the GLOBAL minimum of the potential: "]*)
min1=NMinimize[Re[veff1],{h,s},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet;

lim = 0.5;
hlim = lim;
slim = lim;

min2=NMinimize[{Re[veff2],h>hlim},{h},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet;
min3=NMinimize[{Re[veff3],s>slim},{s},Method->{"DifferentialEvolution","SearchPoints"->acc,"ScalingFactor"->4}] // Quiet;

hmin = h /. min1[[2]];
smin = s /. min1[[2]];

hminlocal = h /. min2[[2]];
sminlocal = s /. min3[[2]];

p1 = -min1[[1]];
p2 = -min2[[1]];
p3 = -min3[[1]];

(*If[plotting==1,Print[{T,Abs[hmin]/T,Abs[smin]/T}]];*)

hGlobalList = Join[hGlobalList,{{T,Abs[hmin]/T}}];
sGlobalList = Join[sGlobalList,{{T,Abs[smin]/T}}];

hLocalList = Join[hLocalList,{{T,Abs[hminlocal]/T}}];
sLocalList = Join[sLocalList,{{T,Abs[sminlocal]/T}}];

p0 = (106.75`+1) \[Pi]^2/90 T^4;

pGlobalList = Join[pGlobalList,{{T,p1/p0}}];
phLocalList = Join[phLocalList,{{T,p2/p0}}];
psLocalList = Join[psLocalList,{{T,p3/p0}}];



(* strict expansion analytically: *)

(* 

V0 = (h^4 \[Lambda])/4+1/4 h^2 s^2 \[Lambda]hs+(s^4 \[Lambda]s)/4+(h^2 \[Mu]hsq)/2+(s^2 \[Mu]ssq)/2;
d1=D[V0,{h,1}]/. s->0
Solve[d1==0,h]
d2=D[V0,{s,1}]/. h->0
Solve[d2==0,s]

h0=(\[ImaginaryI] Sqrt[\[Mu]hsq])/Sqrt[\[Lambda]];
s0 = (\[ImaginaryI] Sqrt[\[Mu]ssq])/Sqrt[\[Lambda]s];
Vh=V0 /. h-> h0 /. s-> 0
Vs=V0 /. s-> s0 /. h-> 0

*)


veff0 = Veff3d[T,\[Mu]sq,Ms,\[Lambda]hs,\[Lambda]s,h,s]; 

h0Strict=veff0[[4]];
s0Strict =veff0[[5]];

VhStrict = veff0[[2]];
VsStrict = veff0[[3]];

p22 = -VhStrict;
p33 = -VsStrict;

hLocalListStrict = Join[hLocalListStrict,{{T,Abs[h0Strict]/T}}];
sLocalListStrict = Join[sLocalListStrict,{{T,Abs[s0Strict]/T}}];

phLocalListStrict = Join[phLocalListStrict,{{T,p22/p0}}];
psLocalListStrict = Join[psLocalListStrict,{{T,p33/p0}}];

If[plotting==1,Print[{T,Abs[hmin]/T,Abs[smin]/T,Abs[h0Strict]/T,Abs[s0Strict]/T}]];


,{T,Tmax, Tmin,-dT}];

r1=ListLinePlot[hGlobalList,PlotStyle->Blue];
r2=ListLinePlot[sGlobalList,PlotStyle->Red];
p11=ListLinePlot[hLocalList,PlotStyle->Cyan,AxesLabel->{"T","v/T"}];
p22=ListLinePlot[sLocalList,PlotStyle->Magenta];
If[plotting==1,Print[Show[p11,p22,r1,r2]]];

u1=ListLinePlot[pGlobalList,PlotStyle->Black];
u2=ListLinePlot[phLocalList, PlotStyle->Blue,AxesLabel->{"T","p/p0"}];
u3=ListLinePlot[psLocalList,PlotStyle->Red];

If[plotting==1,Print[Show[u2,u3,u1]]];

If[plotting==1,Print[Show[u2,u3]]];

phInt=Interpolation[phLocalList];
psInt=Interpolation[psLocalList];

Tguess = 100;
sol=FindRoot[phInt[t]-psInt[t],{t,Tguess}];
(*Print["Tc for transition to EW phase: "];*)
Tc = t /. sol;

(* strict expansion: *)

If[plotting==1,Print["Strict expansion:"]];

p11=ListLinePlot[hLocalListStrict,PlotStyle->Cyan,AxesLabel->{"T","v/T"}];
p22=ListLinePlot[sLocalListStrict,PlotStyle->Magenta];
If[plotting==1,Print[Show[p11,p22]]];

u2=ListLinePlot[phLocalListStrict, PlotStyle->Blue,AxesLabel->{"T","p/p0"}];
u3=ListLinePlot[psLocalListStrict,PlotStyle->Red];

If[plotting==1,Print[Show[u2,u3]]];


phInt=Interpolation[phLocalListStrict];
psInt=Interpolation[psLocalListStrict];

Tguess = 100;
sol=FindRoot[phInt[t]-psInt[t],{t,Tguess}];
(*Print["Tc for transition to EW phase: "];*)
Tc0 = t /. sol;


Return[{Tc,Tc0}];

];


(* test 1: *)
{Ms,\[Lambda]hs,\[Lambda]s} = {103.79,0.7152,1.0}; (* new BM point from Benoits paper!*) (* My current toy algorithm gives Tc = 117.779 while benoits result is 117.73*)

(* test 2: *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {90.0,0.8,1.0};*) (* new BM point from Benoits paper!*) (* Tc = 87.699, Benoit gets 87.79 *)

(* test 1 range: *)
Tmin = 100;
Tmax = 130;
dT = 5;

plotting = 1;

findTc4d[Ms,\[Lambda]hs,\[Lambda]s,Tmin,Tmax,dT,plotting]



(* test 1: *)
{Ms,\[Lambda]hs,\[Lambda]s} = {103.79,0.7152,1.0}; (* new BM point from Benoits paper!*) (* My current toy algorithm gives Tc = 117.779 while benoits result is 117.73*)

(* test 2: *)
(*{Ms,\[Lambda]hs,\[Lambda]s} = {90.0,0.8,1.0};*) (* new BM point from Benoits paper!*) (* Tc = 87.699, Benoit gets 87.79 *)

(* test 1 range: *)
Tmin = 100;
Tmax = 130;
dT = 5;

plotting = 1;

findTc3d[Ms,\[Lambda]hs,\[Lambda]s,Tmin,Tmax,dT,plotting]




