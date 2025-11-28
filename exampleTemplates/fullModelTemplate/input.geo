// Turn on labels
SetFactory("OpenCASCADE");
Mesh.Algorithm = 5;



// Initialise parameters - user coded. 
// Mesh parameters:
meshSize = 0.5;
// Box parameters
Lx = 8; 
Ly = 6;
Lz = 5;//30;
flapWidth = 0.2;
flapHeight = 3.5;
flapBase = flapWidth;
flapStart = 0;
flapLength = Lz;
// Catheter parameters
xc = 2;
yc = 2.5;
r = 1.25;
ri = 0.75;
catheterLength = Lz;

// Hole parameters updated in simulation
alpha = {24};
radius[] = {0.2,0.2,0.2,0.2,0.2,0.2};
zPos[] = {1.2,2.5999999999999996,3.9999999999999996,5.3999999999999995,6.799999999999999,8.2};
number[] = {4,4,4,4,4,4};
offset[] = {0,0,0,0,0,0};


// Start solid

// End solid


// Make box 
boxR = newp; Point(boxR) = {flapBase/2, 0, 0, meshSize}; 
p = newp; Point(p) = {Lx/2, Ly/2, 0, meshSize}; 
p = newp; Point(p) = {flapBase/2, Ly, 0, meshSize}; 
p = newp; Point(p) = {-flapBase/2, Ly, 0, meshSize}; 
p = newp; Point(p) = {-Lx/2, Ly/2, 0, meshSize}; 
boxL = newp; Point(boxL) = {-flapBase/2, 0, 0, meshSize}; 
spline = newl; Spline(spline)={boxR:boxL:1}; 
base = newl; Line(base) = {boxL, boxR};
cl = newcl; Curve Loop(cl) = {spline, base};
boxS = news; Plane Surface(boxS) = {cl}; 

boxFull[] = Extrude { 0,0, Lz-2*0 }{Surface{boxS};};
box = boxFull[1];

//Make catheter 
shuntOuter = newv;
Cylinder(shuntOuter) = {xc, yc, 0, 0, 0, catheterLength, r, 2*Pi};
shuntTip = newv;
Sphere(shuntTip) = {xc, yc, catheterLength, r};
BooleanUnion{Volume{shuntOuter};Delete;}{Volume{shuntTip};Delete;}
shuntInner = newv;
Cylinder(shuntInner) = {xc, yc, 0, 0, 0, catheterLength, ri, 2*Pi};
shuntOuter = box+1;
BooleanDifference{Volume{shuntOuter};Delete;}{Volume{shuntInner};Delete;}

numRings = #radius[];
// Set up each of the hole rings 
For i In {0:numRings-1:1}
    // Set up each of the holes around the ring
    For j In {0:number[i]-1:1}
        z = Lz - zPos[i]; rHole = radius[i]; theta = j * 360 / number[i];
        xPos = 1.05*r * Cos(Pi/180*(alpha + theta + offset[i])); yPos = 1.05 * r * Sin(Pi/180*(alpha + theta + offset[i])); 
        v = newv; 
        If (radius[i] < .10)
	        Cylinder(v) = {xPos+xc, yPos+yc, z, -xPos, -yPos, 0, rHole};
        Else
	        Cone(v) = {xPos+xc, yPos+yc, z, -xPos, -yPos, 0, rHole, 0};
        EndIf
	    BooleanDifference{Volume{shuntOuter};Delete;}{Volume{v}; Delete;}
    EndFor
EndFor
BooleanDifference{Volume{box}; Delete;}{Volume{shuntOuter}; Delete;}
Coherence;
BooleanDifference{Volume{box};Delete;}{Volume{flap}; Delete;}


// Identify groups of surfaces with phyiscal entities
surfs[]=Surface "*";
Physical Surface("walls") = {};
Physical Surface("flap") = {};
Physical Surface("shunt") = {};
Physical Surface("shuntHoles") = {};
// Identify back and front walls
For i In {0:#surfs[]-1:1}
	s = surfs(i);
	Printf("i %g, s[i] %g", i, s);
	bdyline() = Boundary{Surface{s};};
	k=#bdyline();
	Printf("len bdyline %g", k);
	bdy[] = Boundary{Line{bdyline[0]};};
	coords = Point{bdy[0]};
	If (k > 3) // Selects the back and front walls
		Printf("found k > 3, k= %g", k);
		If ((coords[0]-xc)*(coords[0]-xc) + (coords[1]-yc)*(coords[1]-yc) < r*r + .1)
			Physical Surface("shunt",3) += {s};
		Else
			Physical Surface("walls",1) += {s};
		EndIf
	ElseIf (k == 1)
		Printf("found k ==1 k=%g", k);
		If (coords[2]==0)
		    Physical Surface("shuntHoles", 4) += {s};
		Else
		    Physical Surface("shunt", 3) += {s};
		EndIf
	ElseIf (k == 3) //(coords[0] < xc - r - 0.01)
		Physical Surface("flap",1) += {s};
	EndIf
		
EndFor

Physical Volume("fluid") = {box};
Characteristic Length{Point{:}} = meshSize;
Mesh.CharacteristicLengthFromCurvature = 10;
//Characteristic Length{PointsOf{Physical Surface{1};}} = 3*meshSize;

