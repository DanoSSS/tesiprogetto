package com.polimi;

/*
 * resonantBeam_v2.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Apr 22 2025, 16:07 by COMSOL 6.3.0.335. */
public class ModelWrapper {

	  public static Model run() {
		    Model model = ModelUtil.create("Model");

		    model
		         .modelPath("C:\\Users\\danis\\OneDrive - Politecnico di Milano\\MAGISTRIS\\Tesi\\Python_Batch\\java_project\\eclipse-tesi-workspace\\comsol_simulation_data_creation\\src\\main\\resources\\models");

		    model.label("resonantBeamSim_v2.mph");

		    model.param().set("b_length", "60[um]", "beam lenght");
		    model.param().set("b_height", "2[um]", "beam height");
		    model.param().set("air_gap", "1.5 [um]", "air gap between beam and electrode");
		    model.param().group().create("par2");
		    model.param("par2").set("Vbase", "10[V]", "applied DC voltage");
		    model.param("par2").set("Vmax", "35[V]", "maximum voltage");
		    model.param("par2").set("f_0", "1/T_0[Hz]", "Result from study 1 - natural frequency");
		    model.param("par2").set("T_0", "1e-5[s]", "Result from study 1 - natural period");
		    model.param("par2").set("study_type", "1", "study type - 0 for static 1 for dynamic");
		    model.param("par2").set("voltage_type", "0", "voltage_type - 0 for wave, 1 for step");
		    model.param("par2").set("omega_0", "2*pi*f_0[rad/s]", "natural angular velocity");
		    model.param("par2").set("alpha", "4189[Hz]", "damping parameter - alpha");
		    model.param("par2").set("beta", "8.29e-13[s]", "damping parameter - beta");
		    model.param("par2").set("vrel", "-0.3");
		    model.param().label("Geometry e Mesh");
		    model.param("par2").label("Physics");

		    model.component().create("comp1", true);

		    model.component("comp1").geom().create("geom1", 2);

		    model.result().table().create("tbl1", "Table");
		    model.result().table().create("tbl2", "Table");
		    model.result().table().create("tbl3", "Table");
		    model.result().evaluationGroup().create("std3EvgFrq", "EvaluationGroup");
		    model.result().evaluationGroup().create("eg1", "EvaluationGroup");
		    model.result().evaluationGroup().create("std3mpf1", "EvaluationGroup");
		    model.result().evaluationGroup("std3EvgFrq").create("gev1", "EvalGlobal");
		    model.result().evaluationGroup("eg1").create("pev1", "EvalPoint");
		    model.result().evaluationGroup("eg1").create("gev1", "EvalGlobal");
		    model.result().evaluationGroup("eg1").create("gevs1", "EvalGlobalSweep");
		    model.result().evaluationGroup("std3mpf1").create("gev1", "EvalGlobal");

		    model.func().create("wv1", "Wave");
		    model.func().create("step1", "Step");
		    model.func("wv1").label("Sinusoidal Actuation Voltage ");
		    model.func("wv1").set("period", "T_0[s]");
		    model.func("wv1").set("amplitude", "Vbase");
		    model.func("step1").set("location", "0[s]");
		    model.func("step1").set("from", "0[V]");
		    model.func("step1").set("to", "Vbase");
		    model.func("step1").set("smoothactive", false);

		    model.component("comp1").mesh().create("mesh1");

		    model.component("comp1").geom("geom1").lengthUnit("\u00b5m");
		    model.component("comp1").geom("geom1").create("r1", "Rectangle");
		    model.component("comp1").geom("geom1").feature("r1").set("size", new String[]{"b_length", "b_height"});
		    model.component("comp1").geom("geom1").feature("r1").set("pos", new String[]{"-b_length", "air_gap"});
		    model.component("comp1").geom("geom1").feature("r1").set("layername", new String[]{"Layer 1"});
		    model.component("comp1").geom("geom1").feature("r1").setIndex("layer", "b_length/2", 0);
		    model.component("comp1").geom("geom1").feature("r1").set("layerright", true);
		    model.component("comp1").geom("geom1").feature("r1").set("layerbottom", false);
		    model.component("comp1").geom("geom1").create("r2", "Rectangle");
		    model.component("comp1").geom("geom1").feature("r2").set("size", new String[]{"b_length", "air_gap"});
		    model.component("comp1").geom("geom1").feature("r2").set("pos", new String[]{"-b_length", "0"});
		    model.component("comp1").geom("geom1").create("mir1", "Mirror");
		    model.component("comp1").geom("geom1").feature("mir1").set("keep", true);
		    model.component("comp1").geom("geom1").feature("mir1").selection("input").set("r1", "r2");
		    model.component("comp1").geom("geom1").run();
		    model.component("comp1").geom("geom1").run("fin");

		    model.component("comp1").variable().create("var1");
		    model.component("comp1").variable("var1").set("Vt1", "step1(t)");
		    model.component("comp1").variable("var1").set("Vt2", "wv1(t[s])", "time varying voltage");
		    model.component("comp1").variable("var1")
		         .set("Vdc", "if(study_type,if(voltage_type,Vt1,Vt2),vdc)", "switch voltage from static to dynamic");
		    model.component("comp1").variable("var1").set("Vtresh", "withsol('std2',vdc)");

		    model.component("comp1").material().create("mat1", "Common");
		    model.component("comp1").material().create("mat2", "Common");
		    model.component("comp1").material("mat1").selection().set(2, 3, 5, 6);
		    model.component("comp1").material("mat1").propertyGroup()
		         .create("Enu", "Enu", "Young's modulus and Poisson's ratio");
		    model.component("comp1").material("mat2").selection().set(1, 4);

		    model.component("comp1").common().create("free1", "DeformingDomain");
		    model.component("comp1").common().create("mpf1", "ParticipationFactors");
		    model.component("comp1").common("free1").selection().set(1, 4);

		    model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
		    model.component("comp1").physics("solid").selection().set(2, 3, 5, 6);
		    model.component("comp1").physics("solid").feature("lemm1").create("dmp1", "Damping", 2);
		    model.component("comp1").physics("solid").create("fix1", "Fixed", 1);
		    model.component("comp1").physics("solid").feature("fix1").selection().set(3, 18);
		    model.component("comp1").physics().create("es", "Electrostatics", "geom1");
		    model.component("comp1").physics("es").create("term1", "DomainTerminal", 2);
		    model.component("comp1").physics("es").feature("term1").selection().set(2, 3, 5, 6);
		    model.component("comp1").physics("es").create("pot1", "ElectricPotential", 1);
		    model.component("comp1").physics("es").feature("pot1").selection().set(2, 10);
		    model.component("comp1").physics("es").create("ge1", "GlobalEquations", -1);
		    model.component("comp1").physics("es").feature("ge1").set("fieldName", "ODE2");
		    model.component("comp1").physics("es").feature("ge1").set("DependentVariableQuantity", "electricpotential");

		    model.component("comp1").multiphysics().create("eme1", "ElectromechanicalForces", 2);

		    model.component("comp1").mesh("mesh1").create("map1", "Map");
		    model.component("comp1").mesh("mesh1").feature("map1").create("dis1", "Distribution");
		    model.component("comp1").mesh("mesh1").feature("map1").create("dis2", "Distribution");
		    model.component("comp1").mesh("mesh1").feature("map1").feature("dis1").selection().set(1, 3);
		    model.component("comp1").mesh("mesh1").feature("map1").feature("dis2").selection().set(4, 7, 12, 15);

		    model.component("comp1").probe().create("point1", "Point");
		    model.component("comp1").probe("point1").selection().set(7);

		    model.result().table("tbl1").label("Probe Table 1");

		    model.component("comp1").variable("var1").label("Physics Variables");

		    model.component("comp1").view("view1").axis().set("xmin", -59.62211990356445);
		    model.component("comp1").view("view1").axis().set("xmax", 65.9811019897461);
		    model.component("comp1").view("view1").axis().set("ymin", -48.644813537597656);
		    model.component("comp1").view("view1").axis().set("ymax", 41.70136260986328);

		    model.component("comp1").material("mat1").label("Poly-silicon");
		    model.component("comp1").material("mat1").propertyGroup("def").set("density", "2330[kg/m^3]");
		    model.component("comp1").material("mat1").propertyGroup("Enu").set("E", "1E11[Pa]");
		    model.component("comp1").material("mat1").propertyGroup("Enu").set("nu", "0.3");
		    model.component("comp1").material("mat2").label("Air");
		    model.component("comp1").material("mat2").propertyGroup("def")
		         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});

		    model.component("comp1").common("free1").set("smoothingType", "hyperelastic");

		    model.component("comp1").physics("solid").prop("PhysicsSymbols").set("PhysicsSymbols", true);
		    model.component("comp1").physics("solid").prop("Type2D").set("ModeExtension", true);
		    model.component("comp1").physics("solid").feature("lemm1").feature("dmp1").set("alpha_dM", "alpha");
		    model.component("comp1").physics("solid").feature("lemm1").feature("dmp1").set("beta_dK", "beta");
		    model.component("comp1").physics("solid").feature("lemm1").feature("dmp1").active(false);
		    model.component("comp1").physics("es").feature("term1").set("TerminalType", "Voltage");
		    model.component("comp1").physics("es").feature("term1").set("V0", 0);
		    model.component("comp1").physics("es").feature("pot1").set("V0", "Vdc");
		    model.component("comp1").physics("es").feature("ge1").set("name", "vdc");
		    model.component("comp1").physics("es").feature("ge1").set("equation", "vrel-vmid");
		    model.component("comp1").physics("es").feature("ge1").set("description", "DC bias Voltage");

		    model.component("comp1").mesh("mesh1").feature("map1").feature("dis1").set("numelem", 10);
		    model.component("comp1").mesh("mesh1").feature("map1").feature("dis2").set("numelem", 15);
		    model.component("comp1").mesh("mesh1").run();

		    model.component("comp1").probe("point1").label("Relative Displacement at mid cross section");
		    model.component("comp1").probe("point1").set("probename", "vmid");
		    model.component("comp1").probe("point1").set("expr", "v/air_gap");
		    model.component("comp1").probe("point1").set("unit", "1");
		    model.component("comp1").probe("point1").set("descr", "v/air_gap");
		    model.component("comp1").probe("point1").set("method", "summation");
		    model.component("comp1").probe("point1").set("frame", "material");
		    model.component("comp1").probe("point1").set("table", "tbl1");
		    model.component("comp1").probe("point1").set("window", "window2");

		    model.study().create("std1");
		    model.study("std1").create("time", "Transient");
		    model.study("std1").create("tffft", "TimeToFreqFFT");
		    model.study("std1").feature("time").set("useadvanceddisable", true);
		    model.study("std1").feature("time").set("disabledphysics", new String[]{"es/ge1"});
		    model.study("std1").feature("tffft").set("useadvanceddisable", true);
		    model.study("std1").feature("tffft").set("disabledphysics", new String[]{"es/ge1"});
		    model.study().create("std2");
		    model.study("std2").create("stat", "Stationary");
		    model.study().create("std3");
		    model.study("std3").create("eig", "Eigenfrequency");

		    model.batch().create("b1", "Batch");
		    model.batch().create("p1", "Parametric");
		    model.batch().create("p2", "Parametric");
		    model.batch().create("b2", "Batch");
		    model.batch("b1").create("ge1", "Geomseq");
		    model.batch("b1").create("me1", "Meshseq");
		    model.batch("b1").create("so1", "Solutionseq");
		    model.batch("b1").create("en1", "Evalnumericalseq");
		    model.batch("b1").create("ex2", "Exportseq");
		    model.batch("b1").feature("daDef").create("pr1", "Process");
		    model.batch("b1").feature("daDef").create("pr2", "Process");
		    model.batch("b1").feature("daDef").create("pr3", "Process");
		    model.batch("b1").feature("daDef").create("pr4", "Process");
		    model.batch("b1").feature("daDef").create("pr5", "Process");
		    model.batch("b1").feature("daDef").create("pr6", "Process");
		    model.batch("p1").create("jo3", "Jobseq");
		    model.batch("p2").create("so1", "Solutionseq");
		    model.batch("b2").create("nu1", "Numericalseq");
		    model.batch("b1").study("std2");
		    model.batch("p1").study("std2");
		    model.batch("p2").study("std1");
		    model.batch("b2").study("std1");

		    model.sol().create("sol1");
		    model.sol("sol1").attach("std1");
		    model.sol("sol1").create("st1", "StudyStep");
		    model.sol("sol1").create("v1", "Variables");
		    model.sol("sol1").create("t1", "Time");
		    model.sol("sol1").create("su1", "StoreSolution");
		    model.sol("sol1").create("st2", "StudyStep");
		    model.sol("sol1").create("v2", "Variables");
		    model.sol("sol1").create("fft1", "FFT");
		    model.sol().create("sol3");
		    model.sol("sol3").attach("std2");
		    model.sol("sol3").create("st1", "StudyStep");
		    model.sol("sol3").create("v1", "Variables");
		    model.sol("sol3").create("s1", "Stationary");
		    model.sol("sol3").feature("s1").create("p1", "Parametric");
		    model.sol("sol3").feature("s1").create("se1", "Segregated");
		    model.sol("sol3").feature("s1").create("fc1", "FullyCoupled");
		    model.sol("sol3").feature("s1").feature("se1").create("ss1", "SegregatedStep");
		    model.sol("sol3").feature("s1").feature("se1").create("ss2", "SegregatedStep");
		    model.sol("sol3").feature("s1").feature("se1").create("ss3", "SegregatedStep");
		    model.sol("sol3").feature("s1").feature("se1").feature().remove("ssDef");
		    model.sol("sol3").feature("s1").feature().remove("fcDef");
		    model.sol().create("sol4");
		    model.sol("sol4").attach("std3");
		    model.sol("sol4").create("st1", "StudyStep");
		    model.sol("sol4").create("v1", "Variables");
		    model.sol("sol4").create("e1", "Eigenvalue");

		    model.result().dataset().create("dset5", "Solution");
		    model.result().dataset().create("dset6", "Solution");
		    model.result().dataset().create("avh1", "Average");
		    model.result().dataset("dset3").set("probetag", "point1");
		    model.result().dataset("dset4").set("solution", "sol3");
		    model.result().dataset("dset5").set("probetag", "point1");
		    model.result().dataset("dset6").set("solution", "sol4");
		    model.result().dataset("avh1").set("probetag", "point1");
		    model.result().dataset("avh1").set("data", "dset5");
		    model.result().dataset("avh1").selection().geom("geom1", 0);
		    model.result().dataset("avh1").selection().set(7);
		    model.result().numerical().create("gev1", "EvalGlobal");
		    model.result().numerical().create("pev1", "EvalPoint");
		    model.result().numerical().create("pev2", "EvalPoint");
		    model.result().numerical("gev1").set("data", "dset4");
		    model.result().numerical("pev1").set("probetag", "point1");
		    model.result().numerical("pev2").set("data", "dset2");
		    model.result().numerical("pev2").selection().set(7);
		    model.result().create("pg1", "PlotGroup1D");
		    model.result().create("pg2", "PlotGroup2D");
		    model.result().create("pg3", "PlotGroup1D");
		    model.result().create("pg6", "PlotGroup1D");
		    model.result().create("pg7", "PlotGroup1D");
		    model.result().create("pg8", "PlotGroup2D");
		    model.result("pg1").set("data", "dset2");
		    model.result("pg1").create("ptgr1", "PointGraph");
		    model.result("pg1").feature("ptgr1").selection().set(8);
		    model.result("pg1").feature("ptgr1").set("expr", "v");
		    model.result("pg2").set("data", "dset2");
		    model.result("pg2").create("surf1", "Surface");
		    model.result("pg2").feature("surf1").create("def1", "Deform");
		    model.result("pg3").create("ptgr1", "PointGraph");
		    model.result("pg3").feature("ptgr1").selection().set(7);
		    model.result("pg3").feature("ptgr1").set("expr", "abs(v)");
		    model.result("pg6").set("data", "dset4");
		    model.result("pg6").create("glob1", "Global");
		    model.result("pg6").feature("glob1").set("expr", new String[]{"vmid"});
		    model.result("pg7").set("probetag", "window2");
		    model.result("pg7").create("tblp1", "Table");
		    model.result("pg7").feature("tblp1").set("probetag", "point1");
		    model.result("pg8").set("data", "dset6");
		    model.result("pg8").create("surf1", "Surface");
		    model.result().export().create("plot2", "Plot");
		    model.result().export().create("tbl1", "Table");
		    model.result().export().create("plot1", "Plot");
		    model.result().export().create("tbl2", "Table");

		    model.component("comp1").probe("point1").genResult(null);

		    model.result("pg9").tag("pg7");

		    model.study("std1").feature("time").set("tlist", "range(0,T_0/500,3*T_0)");
		    model.study("std1").feature("tffft").set("fftinputstudy", "current");
		    model.study("std1").feature("tffft").set("tunit", "\u00b5s");
		    model.study("std1").feature("tffft").set("fftendtime", "3*T_0");
		    model.study("std1").feature("tffft").set("punit", "kHz");
		    model.study("std1").feature("tffft").set("fftmaxfreq", 3000);
		    model.study("std1").feature("tffft").set("fftscaling", "discrete");
		    model.study("std2").feature("stat").set("useparam", true);
		    model.study("std2").feature("stat").set("sweeptype", "filled");
		    model.study("std2").feature("stat").set("pname", new String[]{"vrel"});
		    model.study("std2").feature("stat").set("plistarr", new String[]{"range(-0.3,-0.01,-0.45)"});
		    model.study("std2").feature("stat").set("punit", new String[]{""});
		    model.study("std3").feature("eig").set("neigsactive", true);
		    model.study("std3").feature("eig").set("shift", "1");
		    model.study("std3").feature("eig").set("shiftactive", true);

		    model.batch("b1").set("batchfileactive", true);
		    model.batch("b1").set("batchfile", "batchmodelStationarySweep.mph");
		    model.batch("b1").set("paramfilename", "index");
		    model.batch("b1").set("clearmesh", true);
		    model.batch("b1").set("clearsol", true);
		    model.batch("b1").feature("daDef")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\backupbatchmodelStationarySweep.mph");
		    model.batch("b1").feature("daDef")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\backupbatchmodelStationarySweep.mph");
		    model.batch("b1").feature("daDef").feature("pr1")
		         .set("cmd", new String[]{"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_copy1\\bin\\win64\\comsolbatch.exe", "-job", "b1", "-pname", "air_gap,b_length,b_height", "-pindex", "0,0,0", "-alivetime", "15", "-inputfile", 
		         "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph\"", "-batchlog", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph.log\"", "-recover"});
		    model.batch("b1").feature("daDef").feature("pr1")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph");
		    model.batch("b1").feature("daDef").feature("pr1")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph");
		    model.batch("b1").feature("daDef").feature("pr1").set("pname", new String[]{"air_gap", "b_length", "b_height"});
		    model.batch("b1").feature("daDef").feature("pr1")
		         .set("plist", "3.000000000000000E-7,2.250000000000000E-5,2.000000000000000E-6");
		    model.batch("b1").feature("daDef").feature("pr1").set("punit", new String[]{"um", "um", "um"});
		    model.batch("b1").feature("daDef").feature("pr1")
		         .set("status", "Mon Apr 28 15:50:19 CEST 2025\nCOMSOL Multiphysics 6.3 (Build: 335) starting in batch mode\nOpening file: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph\nOpen time: 24 s.\nThe input filename C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph will be used as output filename.\nRunning: Batch 1 {b1}\nNumber of vertex elements: 13\nNumber of boundary elements: 260\nMinimum element quality: 1\nMemory: 724/740 752/773\n<---- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} -------------------------------------------------------------------\nStarted at Apr 28, 2025, 3:50:44 PM.\nGeometry shape function: Quadratic serendipity\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 702/740 737/773\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 777/777 802/805\nTime: 7 s.\nPhysical memory: 787 MB\nVirtual memory: 812 MB\nEnded at Apr 28, 2025, 3:50:51 PM.\n----- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} ------------------------------------------------------------------>\n<---- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----\nStarted at Apr 28, 2025, 3:50:51 PM.\nMemory: 786/787 807/812\nSolution time: 0 s.\nPhysical memory: 786 MB\nVirtual memory: 807 MB\nEnded at Apr 28, 2025, 3:50:51 PM.\n----- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} --->\n<---- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ------\nStarted at Apr 28, 2025, 3:50:51 PM.\nContinuation solver\nNonlinear solver\nNumber of degrees of freedom solved for: 11506 (plus 1 internal DOFs).\n  \nContinuation parameter vrel = -0.3.\nContinuation parameter stepsize CMPpcontstep = 0.\nNonsymmetric matrix found.\nScales for dependent variables:\nSpatial Mesh Displacement (comp1.spatial.disp): 6.4e-07\nDisplacement Field (comp1.u): 6.4e-07\nElectric Potential (comp1.V): 75\nGlobal Equations 1 (comp1.ODE2): 2.2e-07\nOrthonormal null-space function used.\nMemory: 846/847 859/860\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 853/857 864/873\n   1     1.5e+06     3.9e+04   0.0000001     1.5e+06    5    1    5    1e-13  2.7e-08\nMemory: 878/879 892/892\n   2     3.7e+03       7e+03   0.0000010     3.7e+03    6    2    7    4e-11  2.3e-14\nMemory: 887/892 899/906\nMemory: 913/917 924/928\n   3     3.7e+03       7e+03   0.0000100     3.7e+03    7    3    9  7.5e-11  2.5e-14\n   4       3e+03       7e+03   0.0001000       3e+03    8    4   11  4.6e-12  2.5e-14\nMemory: 912/926 937/953\n   5     4.6e+02     6.9e+03   0.0010000     4.6e+02    9    5   13  1.8e-12  2.4e-14\nMemory: 915/926 941/953\n   6          59     6.9e+03   0.0100000          60   10    6   15  3.9e-11  2.3e-14\nMemory: 922/926 943/953\n   7         5.8     1.2e+04   0.1000000         6.5   11    7   17  8.7e-11  1.8e-14\nMemory: 908/926 927/953\n   8        0.93       1e+05   0.3672109         1.6   12    8   19  8.5e-11    5e-15\n   9       0.072       3e+05   1.0000000         0.4   13    9   21  2.4e-13  9.5e-12\nMemory: 852/926 877/953\n  10      0.0013     6.1e+02   1.0000000        0.05   14   10   23  7.7e-15  4.2e-11\n  11     1.5e-06      0.0013   1.0000000      0.0014   16   11   25  3.1e-15  7.5e-11\nMemory: 879/926 904/953\n  \nContinuation parameter vrel = -0.31.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08       3e-08   1.0000000     0.00022   21   12   28  2.4e-14  4.1e-11\nMemory: 895/926 920/953\n  \nContinuation parameter vrel = -0.32.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.5e-08       3e-08   1.0000000     0.00024   26   13   31  1.1e-15  4.4e-11\nMemory: 901/926 926/953\n  \nContinuation parameter vrel = -0.33.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 906/926 930/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.3e-08       3e-08   1.0000000     0.00023   31   14   34  1.8e-14  4.1e-11\n  \nContinuation parameter vrel = -0.34.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 898/926 918/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 919/926 941/953\n   1     3.1e-08     2.8e-08   1.0000000     0.00022   36   15   37  6.5e-15  4.1e-11\n  \nContinuation parameter vrel = -0.35.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 907/926 927/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.9e-08       3e-08   1.0000000     0.00022   41   16   40    9e-14  3.8e-11\nMemory: 917/926 939/953\n  \nContinuation parameter vrel = -0.36.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 919/926 941/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.8e-08     2.9e-08   1.0000000     0.00021   46   17   43  3.2e-14    4e-11\n  \nContinuation parameter vrel = -0.37.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 905/926 923/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08     2.7e-08   1.0000000     0.00021   51   18   46  1.1e-14  3.9e-11\n  \nContinuation parameter vrel = -0.38.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 919/926 939/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.6e-08     2.7e-08   1.0000000      0.0002   56   19   49  3.6e-14  3.9e-11\n  \nContinuation parameter vrel = -0.39.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 917/926 937/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.5e-08     2.6e-08   1.0000000      0.0002   61   20   52  3.5e-14  3.5e-11\n  \nContinuation parameter vrel = -0.4.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08       3e-08   1.0000000      0.0002   66   21   55  1.5e-14  3.6e-11\n  \nContinuation parameter vrel = -0.41.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.3e-08       3e-08   1.0000000     0.00019   71   22   58  1.5e-14  3.5e-11\n  \nContinuation parameter vrel = -0.42.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 920/926 936/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 918/926 934/953\n   1     2.2e-08     3.1e-08   1.0000000     0.00019   76   23   61  1.1e-14  3.3e-11\n  \nContinuation parameter vrel = -0.43.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 905/926 935/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.2e-08     3.1e-08   1.0000000     0.00019   81   24   64  2.4e-14  3.4e-11\n  \nContinuation parameter vrel = -0.44.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 899/926 918/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     3.3e-08   1.0000000     0.00018   86   25   67  6.1e-14  3.6e-11\nMemory: 873/926 891/953\n  \nContinuation parameter vrel = -0.45.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 873/926 899/953\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     3.1e-08   1.0000000     0.00018   91   26   70  4.1e-15  3.1e-11\nMemory: 862/926 877/953\nSolution time: 9 s.\nPhysical memory: 926 MB\nVirtual memory: 953 MB\nEnded at Apr 28, 2025, 3:51:00 PM.\n----- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----->\nMemory: 880/926 920/953\nMemory: 886/926 926/953\nRun time: 17 s.\nSaving model: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_0_0_0.mph\nSave time: 2 s.\nTotal time: 43 s.\nMemory: 976/988 986/999");
		    model.batch("b1").feature("daDef").feature("pr2")
		         .set("cmd", new String[]{"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_copy1\\bin\\win64\\comsolbatch.exe", "-job", "b1", "-pname", "air_gap,b_length,b_height", "-pindex", "1,1,1", "-alivetime", "15", "-inputfile", 
		         "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph\"", "-batchlog", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph.log\"", "-recover"});
		    model.batch("b1").feature("daDef").feature("pr2")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph");
		    model.batch("b1").feature("daDef").feature("pr2")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph");
		    model.batch("b1").feature("daDef").feature("pr2").set("pname", new String[]{"air_gap", "b_length", "b_height"});
		    model.batch("b1").feature("daDef").feature("pr2")
		         .set("plist", "8.000000000000000E-7,3.250000000000000E-5,2.500000000000000E-6");
		    model.batch("b1").feature("daDef").feature("pr2").set("punit", new String[]{"um", "um", "um"});
		    model.batch("b1").feature("daDef").feature("pr2")
		         .set("status", "Mon Apr 28 15:51:15 CEST 2025\nCOMSOL Multiphysics 6.3 (Build: 335) starting in batch mode\nOpening file: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph\nOpen time: 29 s.\nThe input filename C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph will be used as output filename.\nRunning: Batch 1 {b1}\nNumber of vertex elements: 13\nNumber of boundary elements: 260\nMinimum element quality: 1\nMemory: 690/701 763/777\n<---- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} -------------------------------------------------------------------\nStarted at Apr 28, 2025, 3:51:44 PM.\nGeometry shape function: Quadratic serendipity\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 694/701 767/777\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 716/716 800/801\nTime: 8 s.\nPhysical memory: 726 MB\nVirtual memory: 808 MB\nEnded at Apr 28, 2025, 3:51:52 PM.\n----- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} ------------------------------------------------------------------>\n<---- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----\nStarted at Apr 28, 2025, 3:51:52 PM.\nMemory: 726/729 803/808\nSolution time: 0 s.\nPhysical memory: 729 MB\nVirtual memory: 806 MB\nEnded at Apr 28, 2025, 3:51:52 PM.\n----- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} --->\n<---- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ------\nStarted at Apr 28, 2025, 3:51:52 PM.\nContinuation solver\nNonlinear solver\nNumber of degrees of freedom solved for: 11506 (plus 1 internal DOFs).\n  \nContinuation parameter vrel = -0.3.\nContinuation parameter stepsize CMPpcontstep = 0.\nNonsymmetric matrix found.\nScales for dependent variables:\nSpatial Mesh Displacement (comp1.spatial.disp): 6.4e-07\nDisplacement Field (comp1.u): 6.4e-07\nElectric Potential (comp1.V): 90\nGlobal Equations 1 (comp1.ODE2): 4.6e-07\nOrthonormal null-space function used.\nMemory: 775/777 851/852\nMemory: 807/810 887/887\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     4.4e+05     2.7e+06   0.0000010     6.7e+05    4    1    4  3.1e-11  1.1e-08\nMemory: 814/817 884/894\nMemory: 838/840 904/907\n   2     9.7e+03       7e+03   0.0000100     9.7e+03    5    2    6  3.4e-11    4e-15\nMemory: 853/870 919/938\n   3     3.7e+03       7e+03   0.0001000     3.7e+03    6    3    8  7.1e-11  2.9e-15\nMemory: 865/870 933/938\n   4     5.4e+02     6.9e+03   0.0010000     5.4e+02    7    4   10  4.6e-11    3e-15\nMemory: 865/871 933/939\n   5          60     6.9e+03   0.0100000          61    8    5   12  1.4e-11    5e-15\nMemory: 871/874 939/942\n   6         5.5       1e+04   0.1000000         6.2    9    6   14    4e-11  3.4e-15\nMemory: 820/874 897/942\n   7        0.33     6.5e+05   1.0000000        0.75   10    7   16  5.6e-12  5.6e-13\nMemory: 821/874 897/942\n   8     0.00036       7e+02   1.0000000       0.075   11    8   18  6.4e-13  6.5e-12\nMemory: 822/874 901/942\n   9       8e-08     1.5e-05   1.0000000     0.00036   13    9   20  9.8e-14  6.1e-11\n  \nContinuation parameter vrel = -0.31.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 826/874 896/942\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.5e-08     4.6e-08   1.0000000     0.00026   18   10   23  3.1e-15  4.1e-11\n  \nContinuation parameter vrel = -0.32.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 845/874 910/942\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.4e-08     4.5e-08   1.0000000     0.00026   23   11   26  1.1e-14    4e-11\nMemory: 846/874 911/942\n  \nContinuation parameter vrel = -0.33.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 847/874 912/942\n   1     3.2e-08     4.6e-08   1.0000000     0.00025   28   12   29  8.3e-14  4.3e-11\n  \nContinuation parameter vrel = -0.34.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 857/874 932/942\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       3e-08     4.2e-08   1.0000000     0.00025   33   13   32  2.6e-14  4.9e-11\n  \nContinuation parameter vrel = -0.35.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 856/874 916/942\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.9e-08     4.4e-08   1.0000000     0.00024   38   14   35  1.8e-14  3.7e-11\n  \nContinuation parameter vrel = -0.36.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 867/874 929/942\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08     4.3e-08   1.0000000     0.00023   43   15   38  2.4e-14  3.4e-11\n  \nContinuation parameter vrel = -0.37.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 862/874 922/942\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.6e-08     4.3e-08   1.0000000     0.00022   48   16   41  5.1e-14  3.3e-11\nMemory: 880/881 941/944\n  \nContinuation parameter vrel = -0.38.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 878/881 939/944\n   1     2.5e-08     4.3e-08   1.0000000     0.00022   53   17   44  1.5e-14  3.5e-11\n  \nContinuation parameter vrel = -0.39.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 875/881 942/944\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08     4.3e-08   1.0000000     0.00021   58   18   47  1.2e-14  3.2e-11\n  \nContinuation parameter vrel = -0.4.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 866/882 923/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.3e-08     4.4e-08   1.0000000     0.00021   63   19   50  4.2e-14  3.1e-11\n  \nContinuation parameter vrel = -0.41.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 865/884 923/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.3e-08     4.2e-08   1.0000000     0.00021   68   20   53  5.6e-15  3.2e-11\n  \nContinuation parameter vrel = -0.42.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 868/884 935/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.2e-08     3.8e-08   1.0000000     0.00021   73   21   56  1.1e-13  2.9e-11\n  \nContinuation parameter vrel = -0.43.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 879/884 939/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 879/884 942/946\n   1     2.1e-08     4.3e-08   1.0000000     0.00021   78   22   59  7.5e-14    3e-11\n  \nContinuation parameter vrel = -0.44.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 870/884 942/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     4.1e-08   1.0000000     0.00022   83   23   62  1.1e-14  3.2e-11\n  \nContinuation parameter vrel = -0.45.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 832/884 891/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     4.1e-08   1.0000000     0.00024   88   24   65  8.3e-14  3.2e-11\nMemory: 836/884 896/946\nSolution time: 9 s.\nPhysical memory: 884 MB\nVirtual memory: 946 MB\nEnded at Apr 28, 2025, 3:52:01 PM.\n----- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----->\nMemory: 855/884 923/946\nMemory: 856/884 923/946\nRun time: 17 s.\nSaving model: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_1_1_1.mph\nSave time: 2 s.\nTotal time: 49 s.\nMemory: 919/932 965/981");

		    return model;
		  }

		  public static Model run2(Model model) {
		    model.batch("b1").feature("daDef").feature("pr3")
		         .set("cmd", new String[]{"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_copy1\\bin\\win64\\comsolbatch.exe", "-job", "b1", "-pname", "air_gap,b_length,b_height", "-pindex", "2,2,2", "-alivetime", "15", "-inputfile", 
		         "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph\"", "-batchlog", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph.log\"", "-recover"});
		    model.batch("b1").feature("daDef").feature("pr3")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph");
		    model.batch("b1").feature("daDef").feature("pr3")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph");
		    model.batch("b1").feature("daDef").feature("pr3").set("pname", new String[]{"air_gap", "b_length", "b_height"});
		    model.batch("b1").feature("daDef").feature("pr3")
		         .set("plist", "1.300000000000000E-6,4.250000000000000E-5,3.000000000000000E-6");
		    model.batch("b1").feature("daDef").feature("pr3").set("punit", new String[]{"um", "um", "um"});
		    model.batch("b1").feature("daDef").feature("pr3")
		         .set("status", "Mon Apr 28 15:52:14 CEST 2025\nCOMSOL Multiphysics 6.3 (Build: 335) starting in batch mode\nOpening file: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph\nOpen time: 23 s.\nThe input filename C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph will be used as output filename.\nRunning: Batch 1 {b1}\nNumber of vertex elements: 13\nNumber of boundary elements: 260\nMinimum element quality: 1\nMemory: 752/752 806/807\n<---- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} -------------------------------------------------------------------\nStarted at Apr 28, 2025, 3:52:38 PM.\nGeometry shape function: Quadratic serendipity\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 749/755 800/807\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 808/812 839/843\nTime: 7 s.\nPhysical memory: 823 MB\nVirtual memory: 848 MB\nEnded at Apr 28, 2025, 3:52:45 PM.\n----- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} ------------------------------------------------------------------>\n<---- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----\nStarted at Apr 28, 2025, 3:52:45 PM.\nMemory: 827/827 848/848\nSolution time: 0 s.\nPhysical memory: 827 MB\nVirtual memory: 848 MB\nEnded at Apr 28, 2025, 3:52:45 PM.\n----- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} --->\n<---- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ------\nStarted at Apr 28, 2025, 3:52:45 PM.\nContinuation solver\nNonlinear solver\nNumber of degrees of freedom solved for: 11506 (plus 1 internal DOFs).\n  \nContinuation parameter vrel = -0.3.\nContinuation parameter stepsize CMPpcontstep = 0.\nNonsymmetric matrix found.\nScales for dependent variables:\nSpatial Mesh Displacement (comp1.spatial.disp): 6.4e-07\nDisplacement Field (comp1.u): 6.4e-07\nElectric Potential (comp1.V): 97\nGlobal Equations 1 (comp1.ODE2): 6.9e-07\nOrthonormal null-space function used.\nMemory: 882/882 903/903\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 886/889 905/909\n   1     1.6e+04     2.1e+06   0.0000010     6.7e+05    4    1    4  8.1e-08  3.2e-11\nMemory: 924/926 941/946\n   2     1.6e+04       7e+03   0.0000100     1.6e+04    5    2    6  4.6e-10  2.4e-15\nMemory: 940/948 964/966\n   3     3.9e+03       7e+03   0.0001000     3.9e+03    6    3    8    1e-10  2.6e-15\nMemory: 942/948 961/966\n   4     5.6e+02     6.9e+03   0.0010000     5.6e+02    7    4   10  2.3e-11  2.5e-15\nMemory: 941/948 958/966\n   5          61     6.9e+03   0.0100000          61    8    5   12  1.4e-10  4.3e-15\n   6         5.5     1.1e+04   0.1000000         6.2    9    6   14  1.3e-10  2.6e-15\nMemory: 929/948 942/966\n   7        0.59     2.1e+05   0.5069371         1.1   10    7   16  7.9e-11  2.5e-15\n   8        0.12     1.3e+05   0.6896518        0.37   11    8   18  1.9e-10  2.6e-15\n   9      0.0018     9.1e+03   1.0000000         0.1   12    9   20  3.4e-12  2.9e-13\nMemory: 930/948 944/966\n  10       1e-07     2.6e-05   1.0000000      0.0018   14   10   22  5.6e-13    6e-12\nMemory: 943/948 958/966\n  \nContinuation parameter vrel = -0.31.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 932/948 976/976\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.8e-08     5.2e-08   1.0000000     0.00032   19   11   25  2.2e-15  4.2e-11\n  \nContinuation parameter vrel = -0.32.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 931/948 958/977\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.3e-08       5e-08   1.0000000      0.0003   24   12   28  7.5e-15  3.6e-11\n  \nContinuation parameter vrel = -0.33.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 944/948 973/977\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 946/948 976/979\n   1     3.1e-08     4.8e-08   1.0000000     0.00028   29   13   31  1.8e-14  3.3e-11\n  \nContinuation parameter vrel = -0.34.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 948/948 983/986\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       3e-08       5e-08   1.0000000     0.00026   34   14   34  1.7e-14  3.4e-11\n  \nContinuation parameter vrel = -0.35.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 940/955 968/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.8e-08     5.2e-08   1.0000000     0.00024   39   15   37  2.5e-14  3.1e-11\n  \nContinuation parameter vrel = -0.36.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 905/955 935/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08     4.9e-08   1.0000000     0.00023   44   16   40  1.7e-14    3e-11\nMemory: 917/955 949/988\n  \nContinuation parameter vrel = -0.37.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 919/955 946/988\n   1     2.5e-08     5.4e-08   1.0000000     0.00022   49   17   43  7.3e-15  3.2e-11\n  \nContinuation parameter vrel = -0.38.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 921/955 950/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08     5.8e-08   1.0000000     0.00021   54   18   46  1.1e-14    3e-11\n  \nContinuation parameter vrel = -0.39.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 914/955 950/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08     6.1e-08   1.0000000     0.00021   59   19   49  7.7e-15  2.9e-11\n  \nContinuation parameter vrel = -0.4.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 910/955 952/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.3e-08     5.8e-08   1.0000000      0.0002   64   20   52    1e-14  2.8e-11\n  \nContinuation parameter vrel = -0.41.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 912/955 936/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.2e-08     5.4e-08   1.0000000      0.0002   69   21   55  1.1e-15  2.7e-11\n  \nContinuation parameter vrel = -0.42.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 909/955 939/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     5.5e-08   1.0000000      0.0002   74   22   58  2.6e-14  2.7e-11\n  \nContinuation parameter vrel = -0.43.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 926/955 953/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 925/955 952/988\n   1     2.1e-08     5.8e-08   1.0000000      0.0002   79   23   61  1.6e-14  2.7e-11\n  \nContinuation parameter vrel = -0.44.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 924/955 959/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     5.6e-08   1.0000000      0.0002   84   24   64  3.4e-14  2.5e-11\n  \nContinuation parameter vrel = -0.45.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 912/955 932/988\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     5.7e-08   1.0000000     0.00021   89   25   67    2e-14  2.5e-11\nSolution time: 9 s.\nPhysical memory: 955 MB\nVirtual memory: 988 MB\nEnded at Apr 28, 2025, 3:52:54 PM.\n----- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----->\nMemory: 917/955 934/988\nMemory: 918/955 934/988\nRun time: 17 s.\nSaving model: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_2_2_2.mph\nSave time: 3 s.\nTotal time: 42 s.\nMemory: 945/955 961/988");
		    model.batch("b1").feature("daDef").feature("pr4")
		         .set("cmd", new String[]{"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_copy1\\bin\\win64\\comsolbatch.exe", "-job", "b1", "-pname", "air_gap,b_length,b_height", "-pindex", "3,3,3", "-alivetime", "15", "-inputfile", 
		         "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph\"", "-batchlog", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph.log\"", "-recover"});
		    model.batch("b1").feature("daDef").feature("pr4")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph");
		    model.batch("b1").feature("daDef").feature("pr4")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph");
		    model.batch("b1").feature("daDef").feature("pr4").set("pname", new String[]{"air_gap", "b_length", "b_height"});
		    model.batch("b1").feature("daDef").feature("pr4")
		         .set("plist", "1.800000000000000E-6,5.250000000000000E-5,3.500000000000000E-6");
		    model.batch("b1").feature("daDef").feature("pr4").set("punit", new String[]{"um", "um", "um"});
		    model.batch("b1").feature("daDef").feature("pr4")
		         .set("status", "Mon Apr 28 15:53:09 CEST 2025\nCOMSOL Multiphysics 6.3 (Build: 335) starting in batch mode\nOpening file: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph\nOpen time: 26 s.\nThe input filename C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph will be used as output filename.\nRunning: Batch 1 {b1}\n<---- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} -------------------------------------------------------------------\nStarted at Apr 28, 2025, 3:53:35 PM.\nGeometry shape function: Quadratic serendipity\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 790/790 848/848\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 846/846 902/902\nTime: 8 s.\nPhysical memory: 862 MB\nVirtual memory: 917 MB\nEnded at Apr 28, 2025, 3:53:42 PM.\n----- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} ------------------------------------------------------------------>\n<---- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----\nStarted at Apr 28, 2025, 3:53:42 PM.\nMemory: 858/862 910/917\nSolution time: 0 s.\nPhysical memory: 858 MB\nVirtual memory: 911 MB\nEnded at Apr 28, 2025, 3:53:43 PM.\n----- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} --->\n<---- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ------\nStarted at Apr 28, 2025, 3:53:43 PM.\nContinuation solver\nNonlinear solver\nNumber of degrees of freedom solved for: 11506 (plus 1 internal DOFs).\n  \nContinuation parameter vrel = -0.3.\nContinuation parameter stepsize CMPpcontstep = 0.\nNonsymmetric matrix found.\nScales for dependent variables:\nSpatial Mesh Displacement (comp1.spatial.disp): 6.4e-07\nDisplacement Field (comp1.u): 6.4e-07\nElectric Potential (comp1.V): 1e+02\nGlobal Equations 1 (comp1.ODE2): 9.3e-07\nOrthonormal null-space function used.\nMemory: 863/865 920/934\nMemory: 902/902 965/965\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e+05     1.7e+06   0.0000010     6.7e+05    4    1    4  2.7e-10  2.8e-09\nMemory: 916/919 968/974\nMemory: 950/959 999/1011\n   2     2.2e+04       7e+03   0.0000100     2.2e+04    5    2    6  5.9e-09  2.4e-15\nMemory: 954/959 1005/1011\nMemory: 977/979 1027/1027\n   3     4.1e+03       7e+03   0.0001000     4.1e+03    6    3    8  4.8e-09  4.4e-15\nMemory: 961/979 1022/1027\n   4     5.8e+02     6.9e+03   0.0010000     5.8e+02    7    4   10  6.9e-10  2.4e-15\nMemory: 961/979 1004/1027\n   5          61     6.9e+03   0.0100000          61    8    5   12  1.1e-08  3.8e-15\n   6         5.5     1.2e+04   0.1000000         6.2    9    6   14  2.7e-09  2.4e-15\nMemory: 968/979 1026/1027\n   7         1.1       1e+05   0.3209618         1.6   10    7   16  2.4e-09  2.4e-15\n   8         0.3     1.4e+05   0.5071974         0.6   11    8   18  2.4e-10  3.1e-15\n   9       0.011     6.9e+04   1.0000000        0.21   12    9   20  2.5e-12  7.4e-13\nMemory: 959/980 1002/1027\n  10     1.2e-05      0.0074   1.0000000       0.011   14   10   22  2.9e-13    1e-11\nMemory: 978/980 1023/1027\n  \nContinuation parameter vrel = -0.31.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 979/981 1023/1027\n   1     2.4e-08     6.6e-08   1.0000000     0.00047   19   11   25  2.1e-14  4.4e-11\n  \nContinuation parameter vrel = -0.32.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 967/981 1054/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.3e-08     6.3e-08   1.0000000     0.00032   24   12   28  4.3e-15  3.1e-11\n  \nContinuation parameter vrel = -0.33.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 979/981 1047/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.1e-08     6.6e-08   1.0000000     0.00028   29   13   31  5.6e-14  3.1e-11\nMemory: 980/983 1048/1055\n  \nContinuation parameter vrel = -0.34.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 970/983 1053/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.9e-08     6.6e-08   1.0000000     0.00026   34   14   34    7e-15    3e-11\n  \nContinuation parameter vrel = -0.35.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 911/983 985/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08     6.3e-08   1.0000000     0.00024   39   15   37  2.2e-14  2.9e-11\nMemory: 916/983 988/1055\n  \nContinuation parameter vrel = -0.36.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 926/983 1005/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.6e-08       7e-08   1.0000000     0.00023   44   16   40  7.2e-14  2.9e-11\n  \nContinuation parameter vrel = -0.37.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 925/983 994/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 945/983 1017/1055\n   1     2.5e-08     6.1e-08   1.0000000     0.00022   49   17   43  9.6e-15  2.8e-11\n  \nContinuation parameter vrel = -0.38.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 931/983 999/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08     6.1e-08   1.0000000     0.00021   54   18   46    1e-14  2.6e-11\n  \nContinuation parameter vrel = -0.39.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 948/983 1015/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.3e-08     6.5e-08   1.0000000     0.00021   59   19   49  6.2e-14  2.5e-11\nMemory: 950/983 1017/1055\n  \nContinuation parameter vrel = -0.4.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 944/983 1024/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.2e-08     6.1e-08   1.0000000      0.0002   64   20   52  4.5e-14  2.5e-11\n  \nContinuation parameter vrel = -0.41.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 941/983 1003/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.2e-08     5.8e-08   1.0000000      0.0002   69   21   55  2.2e-15  2.5e-11\nMemory: 956/983 1020/1055\n  \nContinuation parameter vrel = -0.42.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 952/983 1022/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     6.1e-08   1.0000000      0.0002   74   22   58  6.2e-14  2.5e-11\n  \nContinuation parameter vrel = -0.43.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 956/983 1020/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     5.9e-08   1.0000000      0.0002   79   23   61  9.4e-15  2.4e-11\n  \nContinuation parameter vrel = -0.44.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 943/983 1004/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     5.8e-08   1.0000000      0.0002   84   24   64  2.6e-14  2.3e-11\n  \nContinuation parameter vrel = -0.45.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 956/983 1020/1055\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     5.7e-08   1.0000000      0.0002   89   25   67    2e-14  2.3e-11\nMemory: 938/983 999/1055\nSolution time: 10 s.\nPhysical memory: 983 MB\nVirtual memory: 1055 MB\nEnded at Apr 28, 2025, 3:53:53 PM.\n----- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----->\nMemory: 942/983 1001/1055\nRun time: 19 s.\nSaving model: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_3_3_3.mph\nSave time: 3 s.\nTotal time: 47 s.\nMemory: 970/983 1024/1055");
		    model.batch("b1").feature("daDef").feature("pr5")
		         .set("cmd", new String[]{"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_copy1\\bin\\win64\\comsolbatch.exe", "-job", "b1", "-pname", "air_gap,b_length,b_height", "-pindex", "4,4,4", "-alivetime", "15", "-inputfile", 
		         "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph\"", "-batchlog", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph.log\"", "-recover"});
		    model.batch("b1").feature("daDef").feature("pr5")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph");
		    model.batch("b1").feature("daDef").feature("pr5")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph");
		    model.batch("b1").feature("daDef").feature("pr5").set("pname", new String[]{"air_gap", "b_length", "b_height"});
		    model.batch("b1").feature("daDef").feature("pr5")
		         .set("plist", "2.300000000000000E-6,6.250000000000000E-5,4.000000000000000E-6");
		    model.batch("b1").feature("daDef").feature("pr5").set("punit", new String[]{"um", "um", "um"});
		    model.batch("b1").feature("daDef").feature("pr5")
		         .set("status", "Mon Apr 28 15:54:07 CEST 2025\nCOMSOL Multiphysics 6.3 (Build: 335) starting in batch mode\nOpening file: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph\nOpen time: 19 s.\nThe input filename C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph will be used as output filename.\nRunning: Batch 1 {b1}\nNumber of vertex elements: 13\nNumber of boundary elements: 260\nMinimum element quality: 1\nMemory: 758/768 782/793\n<---- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} -------------------------------------------------------------------\nStarted at Apr 28, 2025, 3:54:26 PM.\nGeometry shape function: Quadratic serendipity\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 762/768 785/793\nRunning on AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 811/811 832/832\nTime: 5 s.\nPhysical memory: 819 MB\nVirtual memory: 838 MB\nEnded at Apr 28, 2025, 3:54:32 PM.\n----- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} ------------------------------------------------------------------>\n<---- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----\nStarted at Apr 28, 2025, 3:54:32 PM.\nMemory: 816/819 833/838\nSolution time: 0 s.\nPhysical memory: 817 MB\nVirtual memory: 834 MB\nEnded at Apr 28, 2025, 3:54:32 PM.\n----- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} --->\n<---- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ------\nStarted at Apr 28, 2025, 3:54:32 PM.\nContinuation solver\nNonlinear solver\nNumber of degrees of freedom solved for: 11506 (plus 1 internal DOFs).\n  \nContinuation parameter vrel = -0.3.\nContinuation parameter stepsize CMPpcontstep = 0.\nNonsymmetric matrix found.\nScales for dependent variables:\nSpatial Mesh Displacement (comp1.spatial.disp): 6.4e-07\nDisplacement Field (comp1.u): 6.4e-07\nElectric Potential (comp1.V): 1e+02\nGlobal Equations 1 (comp1.ODE2): 1.2e-06\nOrthonormal null-space function used.\nMemory: 874/874 893/893\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 893/896 910/914\n   1     3.3e+05     1.4e+06   0.0000010     6.7e+05    4    1    4  2.3e-10  3.6e-09\nMemory: 908/914 921/930\nMemory: 920/924 934/940\n   2     2.7e+04       7e+03   0.0000100     2.7e+04    5    2    6  1.8e-10  2.5e-15\nMemory: 930/932 952/952\n   3     4.2e+03       7e+03   0.0001000     4.2e+03    6    3    8  1.4e-11  2.6e-15\nMemory: 930/936 947/953\n   4     5.8e+02     6.9e+03   0.0010000     5.8e+02    7    4   10  8.7e-10  2.8e-15\n   5          61     6.8e+03   0.0100000          62    8    5   12  1.2e-10  2.7e-15\n   6         5.5     1.3e+04   0.1000000         6.2    9    6   14  3.5e-10  2.4e-15\nMemory: 924/936 935/953\n   7         1.5     6.2e+04   0.2293673           2   10    7   16  3.7e-10  2.7e-15\n   8        0.46     1.4e+05   0.4281065        0.79   11    8   18  2.3e-09  2.8e-15\nMemory: 938/940 970/971\n   9       0.024     1.5e+05   1.0000000        0.27   12    9   20  3.6e-12  1.3e-12\n  10     9.8e-05       0.085   1.0000000       0.025   14   10   22  7.8e-13  1.4e-11\nMemory: 941/941 970/971\n  \nContinuation parameter vrel = -0.31.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 944/944 980/980\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.8e-09       7e-08   1.0000000     0.00072   19   11   25    2e-12  6.5e-12\n  \nContinuation parameter vrel = -0.32.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 934/951 960/980\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     3.4e-08       7e-08   1.0000000     0.00033   24   12   28  2.2e-14    3e-11\n  \nContinuation parameter vrel = -0.33.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 951/951 978/980\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       3e-08     6.9e-08   1.0000000     0.00029   29   13   31  1.2e-14  2.8e-11\nMemory: 948/951 975/980\n  \nContinuation parameter vrel = -0.34.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 949/952 976/980\n   1     2.8e-08     6.8e-08   1.0000000     0.00026   34   14   34  1.3e-13  2.7e-11\n  \nContinuation parameter vrel = -0.35.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 952/955 979/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08     6.3e-08   1.0000000     0.00024   39   15   37  5.2e-14  2.7e-11\n  \nContinuation parameter vrel = -0.36.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 937/955 978/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.6e-08     6.3e-08   1.0000000     0.00023   44   16   40    2e-16  2.6e-11\n  \nContinuation parameter vrel = -0.37.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 940/955 964/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.5e-08     6.4e-08   1.0000000     0.00022   49   17   43  2.1e-13  2.6e-11\n  \nContinuation parameter vrel = -0.38.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 951/955 977/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08     6.1e-08   1.0000000     0.00021   54   18   46  1.3e-13  2.6e-11\n  \nContinuation parameter vrel = -0.39.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 951/955 978/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.3e-08     6.6e-08   1.0000000     0.00021   59   19   49  6.4e-14  2.4e-11\nMemory: 951/955 977/982\n  \nContinuation parameter vrel = -0.4.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 901/955 932/982\n   1     2.2e-08     6.4e-08   1.0000000      0.0002   64   20   52    7e-14  2.3e-11\n  \nContinuation parameter vrel = -0.41.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 899/955 936/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     6.4e-08   1.0000000      0.0002   69   21   55  6.1e-14  2.3e-11\n  \nContinuation parameter vrel = -0.42.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 896/955 937/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.1e-08     6.6e-08   1.0000000      0.0002   74   22   58    6e-14  2.5e-11\n  \nContinuation parameter vrel = -0.43.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 897/955 926/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     6.8e-08   1.0000000     0.00019   79   23   61  3.5e-14  2.2e-11\n  \nContinuation parameter vrel = -0.44.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 910/955 951/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     7.3e-08   1.0000000     0.00019   84   24   64    4e-14  2.5e-11\n  \nContinuation parameter vrel = -0.45.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 926/955 958/982\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     7.8e-08   1.0000000      0.0002   89   25   67  1.1e-13  2.2e-11\nMemory: 925/955 957/982\nSolution time: 8 s.\nPhysical memory: 955 MB\nVirtual memory: 982 MB\nEnded at Apr 28, 2025, 3:54:40 PM.\n----- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----->\nMemory: 915/955 943/982\nMemory: 916/955 943/982\nRun time: 14 s.\nSaving model: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep_4_4_4.mph\nSave time: 2 s.\nTotal time: 35 s.\nMemory: 952/955 973/982");
		    model.batch("b1").feature("daDef").feature("pr6")
		         .set("cmd", new String[]{"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_copy1\\bin\\win64\\comsolbatch.exe", "-job", "b1", "-alivetime", "15", "-inputfile", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph\"", "-batchlog", "\"C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph.log\"", "-recover"});
		    model.batch("b1").feature("daDef").feature("pr6")
		         .set("filename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph");
		    model.batch("b1").feature("daDef").feature("pr6")
		         .set("clientfilename", "C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph");
		    model.batch("b1").feature("daDef").feature("pr6")
		         .set("status", "Tue Apr 29 16:01:21 CEST 2025\nCOMSOL Multiphysics 6.3 (Build: 335) starting in batch mode\nOpening file: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph\nOpen time: 28 s.\nThe input filename C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph will be used as output filename.\nRunning: Batch 1 {b1}\nNumber of vertex elements: 13\nNumber of boundary elements: 260\nMinimum element quality: 1\nMemory: 689/722 721/758\n<---- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} -------------------------------------------------------------------\nStarted at Apr 29, 2025, 4:01:49 PM.\nGeometry shape function: Quadratic serendipity\nRunning on AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 719/722 749/758\nRunning on AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD.\nUsing 1 socket with 4 cores in total on LAPTOP-0H6T75MQ.\nAvailable memory: 14.25 GB.\nMemory: 755/755 775/775\nTime: 5 s.\nPhysical memory: 764 MB\nVirtual memory: 784 MB\nEnded at Apr 29, 2025, 4:01:54 PM.\n----- Compile Equations: Stationary {st1} in Study 2 {std2}/Solution 3 (sol3)\n      {sol3} ------------------------------------------------------------------>\n<---- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----\nStarted at Apr 29, 2025, 4:01:54 PM.\nMemory: 757/764 775/784\nSolution time: 0 s.\nPhysical memory: 757 MB\nVirtual memory: 776 MB\nEnded at Apr 29, 2025, 4:01:54 PM.\n----- Dependent Variables 1 {v1} in Study 2 {std2}/Solution 3 (sol3) {sol3} --->\n<---- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ------\nStarted at Apr 29, 2025, 4:01:54 PM.\nContinuation solver\nNonlinear solver\nNumber of degrees of freedom solved for: 11506 (plus 1 internal DOFs).\n  \nContinuation parameter vrel = -0.3.\nContinuation parameter stepsize CMPpcontstep = 0.\nNonsymmetric matrix found.\nScales for dependent variables:\nSpatial Mesh Displacement (comp1.spatial.disp): 6.4e-07\nDisplacement Field (comp1.u): 6.4e-07\nElectric Potential (comp1.V): 1.1e+02\nGlobal Equations 1 (comp1.ODE2): 1.3e-06\nOrthonormal null-space function used.\nMemory: 828/836 841/849\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     1.4e+06     1.5e+04   0.0000001     1.5e+06    5    1    5  2.5e-11  7.1e-09\nMemory: 853/853 867/867\nMemory: 892/896 906/910\n   2     1.8e+04       7e+03   0.0000010     1.8e+04    6    2    7  1.4e-09  6.4e-15\nMemory: 904/924 918/939\n   3     1.8e+04       7e+03   0.0000100     1.8e+04    7    3    9    1e-09  6.7e-15\n   4     3.9e+03       7e+03   0.0001000     3.9e+03    8    4   11  1.7e-10  6.8e-15\n   5     5.3e+02     6.9e+03   0.0010000     5.3e+02    9    5   13  1.7e-09  6.9e-15\nMemory: 915/929 928/944\n   6          60     5.9e+03   0.0100000          61   10    6   15  1.2e-09  6.4e-15\nMemory: 905/929 915/944\n   7         5.7     2.1e+05   0.1000000         6.3   11    7   17  3.2e-09    6e-15\n   8         0.9     2.7e+06   0.3834293         1.5   12    8   19    1e-09  2.9e-15\nMemory: 918/929 931/944\n   9        0.21     3.4e+06   0.5884349         0.5   13    9   21  7.9e-09  2.6e-15\n  10      0.0087     1.1e+06   1.0000000        0.16   14   10   23  1.1e-11  2.2e-12\nMemory: 907/929 926/944\n  11     1.7e-05       0.035   1.0000000      0.0092   16   11   25  1.5e-11  5.4e-11\n  \nContinuation parameter vrel = -0.31.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 916/929 928/944\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     3.2e-07   1.0000000     0.00031   21   12   28  2.2e-12    1e-10\n  \nContinuation parameter vrel = -0.32.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       3e-08     3.6e-07   1.0000000     0.00026   26   13   31    6e-14  7.7e-11\nMemory: 917/929 928/944\n  \nContinuation parameter vrel = -0.33.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.8e-08     3.6e-07   1.0000000     0.00025   31   14   34  2.2e-12  7.5e-11\n  \nContinuation parameter vrel = -0.34.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 918/929 929/944\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.7e-08     3.9e-07   1.0000000     0.00024   36   15   37  2.5e-12    7e-11\n  \nContinuation parameter vrel = -0.35.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 917/929 929/944\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.5e-08     3.7e-07   1.0000000     0.00023   41   16   40    1e-12  6.9e-11\nMemory: 864/929 896/946\n  \nContinuation parameter vrel = -0.36.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     2.4e-08     3.7e-07   1.0000000     0.00022   46   17   43  2.4e-12  1.1e-10\nMemory: 876/929 906/946\n  \nContinuation parameter vrel = -0.37.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 892/929 921/946\n   1     2.3e-08     3.6e-07   1.0000000     0.00022   51   18   46  3.5e-13  6.8e-11\n  \nContinuation parameter vrel = -0.38.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 888/929 916/946\n   1     2.2e-08     3.7e-07   1.0000000     0.00021   56   19   49  5.2e-13  6.5e-11\n  \nContinuation parameter vrel = -0.39.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 891/929 919/946\n   1     2.1e-08     3.8e-07   1.0000000     0.00021   61   20   52    3e-12    6e-11\n  \nContinuation parameter vrel = -0.4.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\nMemory: 888/929 917/946\n   1     2.1e-08     3.9e-07   1.0000000     0.00021   66   21   55    4e-12  5.9e-11\n  \nContinuation parameter vrel = -0.41.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     3.8e-07   1.0000000     0.00021   71   22   58  3.5e-12  5.9e-11\n  \nContinuation parameter vrel = -0.42.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1       2e-08     3.9e-07   1.0000000     0.00022   76   23   61  6.3e-13  5.6e-11\n  \nContinuation parameter vrel = -0.43.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     1.9e-08       4e-07   1.0000000     0.00022   81   24   64    2e-12  5.8e-11\n  \nContinuation parameter vrel = -0.44.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     1.9e-08     3.8e-07   1.0000000     0.00024   86   25   67  3.6e-12    5e-11\n  \nContinuation parameter vrel = -0.45.\nContinuation parameter stepsize CMPpcontstep = -0.01.\nMemory: 889/929 917/946\nIter      SolEst      ResEst     Damping    Stepsize #Res #Jac #Sol   LinErr   LinRes\n   1     1.9e-08     3.7e-07   1.0000000     0.00027   91   26   70  7.5e-13  5.1e-11\nMemory: 871/929 897/946\nSolution time: 7 s.\nPhysical memory: 929 MB\nVirtual memory: 952 MB\nEnded at Apr 29, 2025, 4:02:01 PM.\n----- Stationary Solver 1 {s1} in Study 2 {std2}/Solution 3 (sol3) {sol3} ----->\nMemory: 881/929 932/952\nMemory: 882/929 932/952\nRun time: 12 s.\nSaving model: C:\\Users\\danis\\Documents\\COMSOL\\Batch\\batchmodelStationarySweep.mph\nSave time: 2 s.\nTotal time: 42 s.\nMemory: 940/941 956/958");

		    return model;
		  }

		  public static Model run3(Model model) {
		    model.batch("b1").feature("ge1").set("seq", "geom1");
		    model.batch("b1").feature("me1").set("seq", "mesh1");
		    model.batch("b1").feature("so1").set("seq", "sol3");
		    model.batch("b1").feature("en1").set("seq", "gev1");
		    model.batch("b1").feature("en1").set("storeoutput", false);
		    model.batch("b1").feature("en1").set("clear", false);
		    model.batch("b1").feature("en1").set("paramtitle", false);
		    model.batch("b1").feature("ex2").label("Pull-In Voltage Max");
		    model.batch("b1").feature("ex2").set("seq", "tbl2");
		    model.batch("b1").feature("ex2").set("paramfilename", "index");
		    model.batch("b1").run();
		    model.batch("p1").label("Full Batch with sweeps");
		    model.batch("p1").set("pname", new String[]{"air_gap", "b_length", "b_height"});
		    model.batch("p1")
		         .set("plistarr", new String[]{"range(0.3, 0.5,2.4)", "range(22.5, 10, 62.6)", "range(2,0.5,4.1)"});
		    model.batch("p1").set("punit", new String[]{"um", "um", "um"});
		    model.batch("p1").run();
		    model.batch("p2").set("pname", new String[]{"Vbase"});
		    model.batch("p2").set("plistarr", new String[]{"1,5,10,20,40,80,100"});
		    model.batch("p2").set("punit", new String[]{"V"});
		    model.batch("p2").feature("so1").set("seq", "sol1");
		    model.batch("p2").run();
		    model.batch("b2").feature("nu1").set("seq", "gev1");
		    model.batch("b2").run();

		    model.sol("sol1").feature("st1").label("Compile Equations: Time Dependent");
		    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
		    model.sol("sol1").feature("v1")
		         .set("clist", new String[]{"{range(0[s], T_0/500, 3*T_0)}", "3.0000000000000004E-8[s]"});
		    model.sol("sol1").feature("t1").label("Time-Dependent Solver 1.1");
		    model.sol("sol1").feature("t1").set("tlist", "range(0,T_0/500,3*T_0)");
		    model.sol("sol1").feature("t1").set("tout", "tstepsclosest");
		    model.sol("sol1").feature("t1").set("rtol", 0.001);
		    model.sol("sol1").feature("t1").set("timemethod", "genalpha");
		    model.sol("sol1").feature("t1").set("tstepsgenalpha", "intermediate");
		    model.sol("sol1").feature("t1").feature("dDef").label("Direct 1");
		    model.sol("sol1").feature("t1").feature("dDef").set("linsolver", "pardiso");
		    model.sol("sol1").feature("t1").feature("aDef").label("Advanced 1");
		    model.sol("sol1").feature("t1").feature("fcDef").label("Fully Coupled 1");
		    model.sol("sol1").feature("su1").label("Solution Store 1.1");
		    model.sol("sol1").feature("st2").label("Compile Equations: Time to Frequency");
		    model.sol("sol1").feature("v2").label("Dependent Variables 2.1");
		    model.sol("sol1").feature("fft1").label("FFT Solver 1.1");
		    model.sol("sol1").feature("fft1").set("tunit", "\u00b5s");
		    model.sol("sol1").feature("fft1").set("fftendtime", "3*T_0");
		    model.sol("sol1").feature("fft1").set("fftinputdata", "fftnotperiodic");
		    model.sol("sol1").feature("fft1").set("punit", "kHz");
		    model.sol("sol1").feature("fft1").set("fftmaxfreq", 3000);
		    model.sol("sol1").feature("fft1").set("fftscaling", "discrete");

		    model.study("std1").runNoGen();

		    model.sol("sol3").feature("st1").label("Compile Equations: Stationary");
		    model.sol("sol3").feature("v1").label("Dependent Variables 1.1");
		    model.sol("sol3").feature("v1").set("clistctrl", new String[]{"p1"});
		    model.sol("sol3").feature("v1").set("cname", new String[]{"vrel"});
		    model.sol("sol3").feature("v1").set("clist", new String[]{"range(-0.3,-0.01,-0.45)"});
		    model.sol("sol3").feature("v1").feature("comp1_spatial_disp").set("scalemethod", "manual");
		    model.sol("sol3").feature("v1").feature("comp1_spatial_disp").set("scaleval", "1e-2*6.403780133639817E-5");
		    model.sol("sol3").feature("v1").feature("comp1_u").set("scalemethod", "manual");
		    model.sol("sol3").feature("v1").feature("comp1_u").set("scaleval", "1e-2*6.403780133639817E-5");
		    model.sol("sol3").feature("s1").label("Stationary Solver 1.1");
		    model.sol("sol3").feature("s1").set("probesel", "none");
		    model.sol("sol3").feature("s1").feature("dDef").label("Direct 1");
		    model.sol("sol3").feature("s1").feature("aDef").label("Advanced 1");
		    model.sol("sol3").feature("s1").feature("aDef").set("cachepattern", true);
		    model.sol("sol3").feature("s1").feature("p1").label("Parametric 1.1");
		    model.sol("sol3").feature("s1").feature("p1").set("sweeptype", "filled");
		    model.sol("sol3").feature("s1").feature("p1").set("pname", new String[]{"vrel"});
		    model.sol("sol3").feature("s1").feature("p1").set("plistarr", new String[]{"range(-0.3,-0.01,-0.45)"});
		    model.sol("sol3").feature("s1").feature("p1").set("punit", new String[]{""});
		    model.sol("sol3").feature("s1").feature("p1").set("uselsqdata", false);
		    model.sol("sol3").feature("s1").feature("se1").label("Segregated 1.1");
		    model.sol("sol3").feature("s1").feature("se1").feature("ss1").active(false);
		    model.sol("sol3").feature("s1").feature("se1").feature("ss1").label("Electric Potential");
		    model.sol("sol3").feature("s1").feature("se1").feature("ss1")
		         .set("segvar", new String[]{"comp1_V", "comp1_ODE2"});
		    model.sol("sol3").feature("s1").feature("se1").feature("ss2").active(false);
		    model.sol("sol3").feature("s1").feature("se1").feature("ss2").label("Displacement Field");
		    model.sol("sol3").feature("s1").feature("se1").feature("ss2").set("segvar", new String[]{"comp1_u"});
		    model.sol("sol3").feature("s1").feature("se1").feature("ss3").active(false);
		    model.sol("sol3").feature("s1").feature("se1").feature("ss3").label("Spatial Mesh Displacement");
		    model.sol("sol3").feature("s1").feature("se1").feature("ss3").set("segvar", new String[]{"comp1_spatial_disp"});
		    model.sol("sol3").feature("s1").feature("fc1").active(true);
		    model.sol("sol3").feature("s1").feature("fc1").label("Fully Coupled 1.1");
		    model.sol("sol3").feature("s1").feature("fc1").set("dtech", "hnlin");

		    model.study("std2").runNoGen();

		    model.sol("sol4").feature("st1").label("Compile Equations: Eigenfrequency");
		    model.sol("sol4").feature("v1").label("Dependent Variables 1.1");
		    model.sol("sol4").feature("v1").feature("comp1_spatial_disp").set("scalemethod", "manual");
		    model.sol("sol4").feature("v1").feature("comp1_spatial_disp").set("scaleval", "1e-2*1.2005103081606586E-4");
		    model.sol("sol4").feature("v1").feature("comp1_u").set("scalemethod", "manual");
		    model.sol("sol4").feature("v1").feature("comp1_u").set("scaleval", "1e-2*1.2005103081606586E-4");
		    model.sol("sol4").feature("e1").label("Eigenvalue Solver 1.1");
		    model.sol("sol4").feature("e1").set("control", "user");
		    model.sol("sol4").feature("e1").set("shift", "1");
		    model.sol("sol4").feature("e1").set("linpmethod", "sol");
		    model.sol("sol4").feature("e1").set("storelinpoint", true);
		    model.sol("sol4").feature("e1").set("eigvfunscale", "maximum");
		    model.sol("sol4").feature("e1").set("eigvfunscaleparam", 1);
		    model.sol("sol4").feature("e1").set("filtereigexpression", new String[]{"real(freq)+1e-6>0"});
		    model.sol("sol4").feature("e1").set("filtereigdescription", new String[]{"Damped natural frequency"});
		    model.sol("sol4").feature("e1").feature("dDef").label("Direct 1");
		    model.sol("sol4").feature("e1").feature("aDef").label("Advanced 1");
		    model.sol("sol4").feature("e1").feature("aDef").set("cachepattern", true);

		    model.study("std3").runNoGen();

		    model.result().dataset("dset3").label("Probe Solution 3");
		    model.result().dataset("dset5").label("Probe Solution 5");
		    model.result().dataset("dset5").set("frametype", "material");
		    model.result().dataset("avh1").set("data", "dset5");
		    model.result().dataset().remove("dset7");
		    model.result().numerical("gev1").label("Pull-In Voltage");
		    model.result().numerical("gev1").set("table", "tbl3");
		    model.result().numerical("gev1").set("expr", new String[]{"vdc"});
		    model.result().numerical("gev1").set("unit", new String[]{"V"});
		    model.result().numerical("gev1").set("descr", new String[]{""});
		    model.result().numerical("gev1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().numerical("gev1").set("dataseries", "maximum");
		    model.result().numerical("gev1").set("includeparam", true);
		    model.result().numerical("pev1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().numerical("pev2").label("DIsplacement at mid-cross section");
		    model.result().numerical("pev2").set("expr", new String[]{"v"});
		    model.result().numerical("pev2").set("unit", new String[]{"\u00b5m"});
		    model.result().numerical("pev2").set("descr", new String[]{"Displacement field, Y-component"});
		    model.result().numerical("pev2")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().numerical("gev1").setResult();
		    model.result().evaluationGroup("std3EvgFrq").label("Eigenfrequencies (Study 3)");
		    model.result().evaluationGroup("std3EvgFrq").set("data", "dset6");
		    model.result().evaluationGroup("std3EvgFrq").set("looplevelinput", new String[]{"manual"});
		    model.result().evaluationGroup("std3EvgFrq").set("looplevel", new String[]{"1, 2, 3, 4, 5, 6"});
		    model.result().evaluationGroup("std3EvgFrq").feature("gev1")
		         .set("expr", new String[]{"2*pi*freq", "imag(freq)/abs(freq)", "abs(freq)/imag(freq)/2"});
		    model.result().evaluationGroup("std3EvgFrq").feature("gev1").set("unit", new String[]{"rad/s", "1", "1"});
		    model.result().evaluationGroup("std3EvgFrq").feature("gev1")
		         .set("descr", new String[]{"Angular frequency", "Damping ratio", "Quality factor"});
		    model.result().evaluationGroup("std3EvgFrq").feature("gev1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().evaluationGroup("std3EvgFrq").run();
		    model.result().evaluationGroup("eg1").feature("pev1").set("data", "dset1");
		    model.result().evaluationGroup("eg1").feature("pev1").set("looplevelinput", new String[]{"all"});
		    model.result().evaluationGroup("eg1").feature("pev1").set("expr", new String[]{"abs(v)"});
		    model.result().evaluationGroup("eg1").feature("pev1").set("unit", new String[]{"\u00b5m"});
		    model.result().evaluationGroup("eg1").feature("pev1")
		         .set("descr", new String[]{"Absolute displacement at mid cross section"});
		    model.result().evaluationGroup("eg1").feature("pev1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().evaluationGroup("eg1").feature("pev1").selection().set(7);
		    model.result().evaluationGroup("eg1").feature("gev1").set("data", "dset4");
		    model.result().evaluationGroup("eg1").feature("gev1").set("looplevelinput", new String[]{"all"});
		    model.result().evaluationGroup("eg1").feature("gev1").set("expr", new String[]{"vdc"});
		    model.result().evaluationGroup("eg1").feature("gev1").set("unit", new String[]{"V"});
		    model.result().evaluationGroup("eg1").feature("gev1").set("descr", new String[]{""});
		    model.result().evaluationGroup("eg1").feature("gev1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().evaluationGroup("eg1").feature("gev1").set("dataseries", "maximum");
		    model.result().evaluationGroup("eg1").feature("gev1").set("includeparam", true);
		    model.result().evaluationGroup("eg1").run();
		    model.result().evaluationGroup("std3mpf1").label("Participation Factors (Study 3)");
		    model.result().evaluationGroup("std3mpf1").set("data", "dset6");
		    model.result().evaluationGroup("std3mpf1").set("looplevelinput", new String[]{"all"});
		    model.result().evaluationGroup("std3mpf1").feature("gev1")
		         .set("expr", new String[]{"mpf1.pfLnormX", "mpf1.pfLnormY", "mpf1.pfLnormZ", "mpf1.pfRnormX", "mpf1.pfRnormY", "mpf1.pfRnormZ", "mpf1.mEffLX", "mpf1.mEffLY", "mpf1.mEffLZ", "mpf1.mEffRX", 
		         "mpf1.mEffRY", "mpf1.mEffRZ"});
		    model.result().evaluationGroup("std3mpf1").feature("gev1")
		         .set("unit", new String[]{"1", "1", "1", "1", "1", "1", "kg", "kg", "kg", "kg*m^2", 
		         "kg*m^2", "kg*m^2"});
		    model.result().evaluationGroup("std3mpf1").feature("gev1")
		         .set("descr", new String[]{"Participation factor, normalized, X-translation", "Participation factor, normalized, Y-translation", "Participation factor, normalized, Z-translation", "Participation factor, normalized, X-rotation", "Participation factor, normalized, Y-rotation", "Participation factor, normalized, Z-rotation", "Effective modal mass, X-translation", "Effective modal mass, Y-translation", "Effective modal mass, Z-translation", "Effective modal mass, X-rotation", 
		         "Effective modal mass, Y-rotation", "Effective modal mass, Z-rotation"});
		    model.result().evaluationGroup("std3mpf1").feature("gev1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result().evaluationGroup("std3mpf1").run();
		    model.result("pg1").label("Displacement at mid cross section");
		    model.result("pg1").set("ylabel", "Displacement field, Y-component (m)");
		    model.result("pg1").set("ylabelactive", false);
		    model.result("pg1").feature("ptgr1").set("unit", "m");
		    model.result("pg1").feature("ptgr1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result("pg1").feature("ptgr1").set("linewidth", "preference");
		    model.result("pg2").set("looplevel", new int[]{1484});
		    model.result("pg2").feature("surf1").set("unit", "m");
		    model.result("pg2").feature("surf1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result("pg2").feature("surf1").set("resolution", "normal");
		    model.result("pg2").feature("surf1").feature("def1").set("scaleactive", true);
		    model.result("pg3").set("looplevelinput", new String[]{"manual"});
		    model.result("pg3").set("xlabel", "freq (kHz)");
		    model.result("pg3").set("ylabel", "abs(v) (m)");
		    model.result("pg3").set("xlabelactive", false);
		    model.result("pg3").set("ylabelactive", false);
		    model.result("pg3").feature("ptgr1").set("unit", "m");
		    model.result("pg3").feature("ptgr1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result("pg3").feature("ptgr1").set("linewidth", "preference");
		    model.result("pg6").set("xlabel", "DC bias Voltage (V)");
		    model.result("pg6").set("ylabel", "v / gap");
		    model.result("pg6").set("ylabelactive", true);
		    model.result("pg6").set("showlegends", false);
		    model.result("pg6").set("xlabelactive", false);
		    model.result("pg6").feature("glob1").set("descr", new String[]{""});
		    model.result("pg6").feature("glob1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result("pg6").feature("glob1").set("xdata", "expr");
		    model.result("pg6").feature("glob1").set("xdataexpr", "vdc");
		    model.result("pg6").feature("glob1").set("xdataunit", "V");
		    model.result("pg6").feature("glob1").set("xdatadescr", "DC bias Voltage");
		    model.result("pg6").feature("glob1").set("linewidth", "preference");
		    model.result("pg7").label("Probe Plot Group 7");
		    model.result("pg7").set("xlabel", "Time (s)");
		    model.result("pg7").set("ylabel", "v/air_gap (1), Relative Displacement at mid cross section {avh1}");
		    model.result("pg7").set("windowtitle", "Probe Plot 2");
		    model.result("pg7").set("xlabelactive", false);
		    model.result("pg7").set("ylabelactive", false);
		    model.result("pg8").feature("surf1")
		         .set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x-coordinate"}, {"solid.refpnty", "0", "Reference point for moment computation, y-coordinate"}, {"solid.refpntz", "0", "Reference point for moment computation, z-coordinate"}});
		    model.result("pg8").feature("surf1").set("resolution", "normal");
		    model.result().export("plot2").label("FFT of Vertical Displacement at Mid Cross Section");
		    model.result().export("plot2").set("plotgroup", "pg3");
		    model.result().export("tbl1").label("Evaluation Group - fft e pullin");
		    model.result().export("tbl1").set("source", "evaluationgroup");
		    model.result().export("tbl1").set("evaluationgroup", "eg1");
		    model.result().export("tbl1").set("header", false);
		    model.result().export("plot1").label("Pull-In Voltage");
		    model.result().export("plot1").set("plotgroup", "pg6");
		    model.result().export("plot1").set("plot", "glob1");
		    model.result().export("plot1")
		         .set("filename", "C:\\Users\\danis\\OneDrive - Politecnico di Milano\\MAGISTRIS\\Tesi\\Python_Batch\\java_project\\Progetti_COMSOL\\simulation_data\\pull-in-voltage1.txt");
		    model.result().export("plot1").set("header", false);
		    model.result().export("plot1").set("fullprec", false);
		    model.result().export("plot1").set("ifexists", "append");
		    model.result().export("tbl2").set("table", "tbl2");
		    model.result().export("tbl2")
		         .set("filename", "C:\\Users\\danis\\OneDrive - Politecnico di Milano\\MAGISTRIS\\Tesi\\Python_Batch\\java_project\\Progetti_COMSOL\\simulation_data\\Pull-In-max.txt");
		    model.result().export("tbl2").set("header", false);

		    return model;
		  }
  
		  public static Model run4(Model model) {
			  model.result("pg6").create("ptgr1", "PointGraph");
			  ResultFeature pg6 = model.result("pg6").feature("ptgr1");
			  pg6.set("markerpos", "datapoints");
			  pg6.set("linewidth", "preference");
			  pg6.selection().set(1);
			  pg6.set("xdatasolnumtype", "outer");
			  pg6.set("titletype", "custom");
			  pg6.set("typeintitle", false);
			  pg6.set("descriptionintitle", false);
			  pg6.set("prefixintitle", "Eigenfrequency vs. DC Voltage");
			  pg6.set("linestyle", "none");
			  pg6.set("linemarker", "square");
			  pg6.set("expr", "freq");
			  pg6.set("xdata", "expr");
			  pg6.set("xdataexpr", "Vdc");

			  return model;
		  }
  public static void printFormattedData(String[] headers, double[][] data, int numRowsToPrint) {

      // --- print headers ---
      if (headers != null && headers.length > 0) {
          for (String header : headers) {
              System.out.printf("%-15s ", header);
          }
          System.out.println();

          for (int i = 0; i < headers.length; i++) {
              System.out.print("--------------- ");
          }
          System.out.println();
      } else {
          System.out.println("Attenzione: Intestazioni non fornite o vuote.");
      }

      // row data
      if (data != null && data.length > 0) {

          int rowsToActuallyPrint = Math.min(numRowsToPrint, data.length);
          if (rowsToActuallyPrint > 0 && headers != null && data[0].length != headers.length) {
               System.out.println("Attenzione: Il numero di colonne dati (" + data[0].length + ") non corrisponde al numero di intestazioni (" + headers.length + "). La stampa potrebbe essere disallineata.");
          }


          System.out.println("Stampa delle prime " + rowsToActuallyPrint + " righe di dati:");

          for (int i = 0; i < rowsToActuallyPrint; i++) {

              if (data[i] == null) {
                  System.out.println("Attenzione: Riga " + i + " è null.");
                  continue;
              }

              for (int j = 0; j < data[i].length; j++) {

                  System.out.printf("%-15.6f ", data[i][j]);
              }
              System.out.println();
          }

          if (data.length > numRowsToPrint) {
               System.out.println("... (" + (data.length - numRowsToPrint) + " righe rimanenti non mostrate)");
          }

      } else {
          System.out.println("Nessun dato da stampare.");
      }
  }

  public static void main(String[] args) {
    run();
    /*
    Connessione a COMSOL

    Setup parametri per Simulazione 1

    Esecuzione e salvataggio risultati

    Esecuzione di Simulazioni 2 e 3 dipendenti dalla 1

    Salvataggio e post-processing
    */
  }

}
