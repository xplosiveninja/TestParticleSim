#define _USE_MATH_DEFINES
#define M_PI 3.14159265358979323846

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <errno.h>
#include "device_launch_parameters.h"
#include <thread>
#include <random>
using namespace std;

//Define Dimesions of Grid Space
__constant__ const int X_Dim = 300;
__constant__ const int Y_Dim = 50;
__constant__ const int Z_Dim = 3;
__constant__ const int TotFieldPoints = X_Dim * Y_Dim * Z_Dim * 3;
__constant__ double X_center_offset = (double)(X_Dim - 1) / 2;
__constant__ double Y_center_offset = (double)(Y_Dim - 1) / 2;
__constant__ double Z_center_offset = (double)(Z_Dim - 1) / 2;
__constant__ int X_Field_Map = 3 * Y_Dim * Z_Dim;
__constant__ int Y_Field_Map = 3 * Z_Dim;
__constant__ int Z_Field_Map = 3;
__constant__ const int InterpFrames = 1000;
__constant__ const double InterpInverse = (double)(1 / 1000);

//Define Total Number of Mag Steps
const int Steps = 768;

//Define total number of charges
__constant__ const int TotCharge = 1000000;

//Define my Value for dt
__constant__ double Delta_T = 0.2 / 1000;

//Allocate Flattened Temporary Field
double M_Field_Temp[X_Dim * Y_Dim * Z_Dim * 3];
double E_Field_Temp[X_Dim * Y_Dim * Z_Dim * 3];

//Set the relative grid spacing increment
double Grid_Increment = 0.25;
__constant__ double Grid_Increment_Cuda = 0.25;
__constant__ const double Inverse_Grid_Increment_Cuda = 1 / 0.25;

//Define total number of grid points
int TotField = (X_Dim * Y_Dim * Z_Dim);

//Allocate Flattened Particle specific Magnetic and Electric Field Values
double Flat_E_Field[3 * TotCharge] = {};
double Flat_M_Field[3 * TotCharge] = {};

//Allocate Flattened Sys Location and Velocity Arrays
double Flat_Location_Vector[3 * TotCharge];
double Flat_Velocity_Vector[3 * TotCharge];

//Allocate Flat Array for X,Y,Z for dumb reasons
double X[X_Dim * Y_Dim * Z_Dim];
double Y[X_Dim * Y_Dim * Z_Dim];
double Z[X_Dim * Y_Dim * Z_Dim];

double T1_Field[X_Dim * Y_Dim * Z_Dim * 3] = {};
double T2_Field[X_Dim * Y_Dim * Z_Dim * 3] = {};
double TE1_Field[X_Dim * Y_Dim * Z_Dim * 3] = {};
double TE2_Field[X_Dim * Y_Dim * Z_Dim * 3] = {};

__global__ void BorisMethod(double *Elec_Field, double *Mag_Field, double *LocVector, double *VelVector) {
	int Charge = blockDim.x * blockIdx.x + threadIdx.x;
	if (Charge < TotCharge) {
		double q_m = -2000;

		//Finding relevant location in Arrays for specific particle
		int FieldIndex = Charge * 3;

		//Allocate arrays
		double V_Minus[3];
		double V_Prime[3];
		double V_Plus[3];
		double t_Vector[3];
		double s_Vector[3];
		double CommonConst = q_m * Delta_T * 0.5;

		//Populate t vector and find t magnitude using Half mag step
		for (int i = 0; i < 3; i++) {
			t_Vector[i] = CommonConst * Mag_Field[FieldIndex + i];
		}
		double t_Mag = (t_Vector[0] * t_Vector[0]) + (t_Vector[1] * t_Vector[1]) + (t_Vector[2] * t_Vector[2]);

		//Populate s vector
		for (int i = 0; i < 3; i++) {
			s_Vector[i] = 2 * t_Vector[i] / (1 + t_Mag);
		}

		//Populate V_minus with a half step in E field
		for (int i = 0; i < 3; i++) {
			V_Minus[i] = VelVector[FieldIndex + i] + CommonConst * Elec_Field[FieldIndex + i];
		}

		//Calculate V cross T components
		double V_Min_Cross_T_i = (V_Minus[1] * t_Vector[2]) - (V_Minus[2] * t_Vector[1]);
		double V_Min_Cross_T_j = (V_Minus[2] * t_Vector[0]) - (V_Minus[0] * t_Vector[2]);
		double V_Min_Cross_T_k = (V_Minus[0] * t_Vector[1]) - (V_Minus[1] * t_Vector[0]);

		//Calculate V prime components
		V_Prime[0] = V_Minus[0] + V_Min_Cross_T_i;
		V_Prime[1] = V_Minus[1] + V_Min_Cross_T_j;
		V_Prime[2] = V_Minus[2] + V_Min_Cross_T_k;

		//Calculate V prime cross S components
		double V_Pri_Cross_S_i = (V_Prime[1] * s_Vector[2]) - (V_Prime[2] * s_Vector[1]);
		double V_Pri_Cross_S_j = (V_Prime[2] * s_Vector[0]) - (V_Prime[0] * s_Vector[2]);
		double V_Pri_Cross_S_k = (V_Prime[0] * s_Vector[1]) - (V_Prime[1] * s_Vector[0]);

		//Calculate V plus components
		V_Plus[0] = V_Minus[0] + V_Pri_Cross_S_i;
		V_Plus[1] = V_Minus[1] + V_Pri_Cross_S_j;
		V_Plus[2] = V_Minus[2] + V_Pri_Cross_S_k;

		//Calculate final Velocity
		for (int i = 0; i < 3; i++) {
			VelVector[FieldIndex + i] = V_Plus[i] + CommonConst * Elec_Field[FieldIndex + i];
		}

		//Add Final Velocity to Location
		for (int i = 0; i < 3; i++) {
			LocVector[FieldIndex + i] += VelVector[FieldIndex + i] * Delta_T;
		}

		//Cyclic Boundary Conditions
		if (LocVector[FieldIndex + 0] >= Grid_Increment_Cuda * X_center_offset) {
			//LocVector[FieldIndex + 0] -= Grid_Increment_Cuda * 2 * X_center_offset;
			//LocVector[FieldIndex + 0] -= VelVector[FieldIndex + 0] * Delta_T;
			//VelVector[FieldIndex + 0] = -VelVector[FieldIndex + 0];
		}
		if (LocVector[FieldIndex + 1] >= Grid_Increment_Cuda * Y_center_offset) {
			LocVector[FieldIndex + 1] -= Grid_Increment_Cuda * 2 * Y_center_offset;
		}
		if (LocVector[FieldIndex + 2] >= Grid_Increment_Cuda * Z_center_offset) {
			LocVector[FieldIndex + 2] -= Grid_Increment_Cuda * 2 * Z_center_offset;
		}
		if (LocVector[FieldIndex + 0] <= -(Grid_Increment_Cuda) * X_center_offset) {
			//LocVector[FieldIndex + 0] += Grid_Increment_Cuda * 2 * X_center_offset;
		}
		if (LocVector[FieldIndex + 1] <= -(Grid_Increment_Cuda) * Y_center_offset) {
			LocVector[FieldIndex + 1] += Grid_Increment_Cuda * 2 * Y_center_offset;
		}
		if (LocVector[FieldIndex + 2] <= -(Grid_Increment_Cuda) * Z_center_offset) {
			LocVector[FieldIndex + 2] += Grid_Increment_Cuda * 2 * Z_center_offset;
		}
	}
}

//Fills Grid space with appropriate magnetic field values. Cuurently setup for 100x100x100 field.
__global__ void M_Field_Fill(double *Field_Grid, double* X, double* Y, double* Z, int Tot, double Grid_Increment_Cuda) {
	int Loc = blockDim.x * blockIdx.x + threadIdx.x;
	if (Loc < Tot) {

		//Find Location of Grid Point
		double x = (double)X[Loc] * Grid_Increment_Cuda;
		double y = (double)Y[Loc] * Grid_Increment_Cuda;
		double z = (double)Z[Loc] * Grid_Increment_Cuda;

		//Set Positions of 2 poles of magnet for bottle/dipole
		double Position_1[3] = { (50 * Grid_Increment_Cuda), (50 * Grid_Increment_Cuda), (100.5 * Grid_Increment_Cuda) };
		double Position_2[3] = { (50 * Grid_Increment_Cuda), (50 * Grid_Increment_Cuda), (-0.5 * Grid_Increment_Cuda) };

		//Set mu vector and allocated 2 vectors for grid point distance calculations
		double mu[3] = { 0, 0, 5000000 };
		double Vector_1[3];
		double Vector_2[3];

		//Find Distance in X,Y,Z of grid point from poles
		Vector_1[0] = Position_1[0] - x;
		Vector_1[1] = Position_1[1] - y;
		Vector_1[2] = Position_1[2] - z;
		Vector_2[0] = Position_2[0] - x;
		Vector_2[1] = Position_2[1] - y;
		Vector_2[2] = Position_2[2] - z;

		//Find Magnitudes of total distance
		double Mag_1 = sqrt((Vector_1[0] * Vector_1[0]) + (Vector_1[1] * Vector_1[1]) + (Vector_1[2] * Vector_1[2]));
		double Mag_2 = sqrt((Vector_2[0] * Vector_2[0]) + (Vector_2[1] * Vector_2[1]) + (Vector_2[2] * Vector_2[2]));

		//Allocate B contributions
		double B_1[3];
		double B_2[3];

		//Calculate Final Magnetic Field properties at grid point, At the moment the code only accounts for mu solely in the Z direction because more wasn't needed at the time, still hasn't been needed, and may never be needed
		for (int i = 0; i < 3; i++) {
			B_1[i] = (Vector_1[i] * 3 * (mu[2] * Vector_1[2]) / pow(Mag_1, 5)) - (mu[i] / pow(Mag_1, 3));
			B_2[i] = (Vector_2[i] * 3 * (mu[2] * Vector_2[2]) / pow(Mag_2, 5)) - (mu[i] / pow(Mag_2, 3));
			Field_Grid[(Loc * 3) + i] = (B_1[i] + B_2[i]);
		}
	}
}

//Just some aribtrary function to intitialise particles
void Loc_ini() {
	int Count = 0;
	std::default_random_engine generator;
	std::gamma_distribution<double> distribution(1.5, 30.28);
	for (int i = 0; i < TotCharge; i++) {
		double r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		double r2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		double v = distribution(generator);
		double angle = 2 * M_PI * r;
		double angle2 = 2 * M_PI * r2;
		double Vx = v * cos(angle) * sin(angle2);
		double Vy = v * sin(angle) * sin(angle2);
		double Vz = v * cos(angle2);
		double r_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		double r_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		Flat_Velocity_Vector[(i * 3) + 0] = Vx;
		Flat_Velocity_Vector[(i * 3) + 1] = Vy;
		Flat_Velocity_Vector[(i * 3) + 2] = Vz;
		Flat_Location_Vector[(i * 3) + 0] = (50 + (r_x * 10)) * Grid_Increment;
		Flat_Location_Vector[(i * 3) + 1] = (6 + (r_y * 5)) * Grid_Increment;
		Flat_Location_Vector[(i * 3) + 2] = 0;
	}
}

__global__ void GridFinder(double* CoOrd, double* GridLoc) {
	int Charge = (blockDim.x * blockIdx.x + threadIdx.x);
	if (Charge < TotCharge) {

		//For each dimension, find grid refernce location, has now been expanded to work on cuboid grids
		double X_GridMap = floor(CoOrd[(Charge * 3) + 0] * Inverse_Grid_Increment_Cuda);
		GridLoc[(Charge * 3) + 0] = X_GridMap + X_center_offset;

		double Y_GridMap = floor(CoOrd[(Charge * 3) + 1] * Inverse_Grid_Increment_Cuda);
		GridLoc[(Charge * 3) + 1] = Y_GridMap + Y_center_offset;

		double Z_GridMap = floor(CoOrd[(Charge * 3) + 2] * Inverse_Grid_Increment_Cuda);
		GridLoc[(Charge * 3) + 2] = Z_GridMap + Z_center_offset;
	}
}

__global__ void Interpolation(double *LocVector, double *Mag_Field, double *LocGrid, double *ResultField) {
	int Vec_Comp = (blockDim.x * blockIdx.x + threadIdx.x);
	if (Vec_Comp < TotCharge) {
	//Test to see if Particle is within grid with very generous buffer
		int BaseComponent = Vec_Comp * 3;
		bool proceed = false;
		if (abs(LocVector[(BaseComponent) + 0]) <= (Grid_Increment_Cuda * (X_center_offset))) {
			if (abs(LocVector[(BaseComponent) + 1]) <= (Grid_Increment_Cuda * (Y_center_offset))) {
				if (abs(LocVector[(BaseComponent) + 2]) <= (Grid_Increment_Cuda * (Z_center_offset))) {
					proceed = true;
				}
			}
		}

		//If particle is inside Grid
		if (proceed == true) {

			//Cast Particle Grid Location reference to integer instead of double to get rid of decimal, Seems dumb to do it here because it is, but it works and I lost 2 weeks of my life to this so it stays
			int Grid_x = (LocGrid[(BaseComponent)+0]);
			int Grid_y = (LocGrid[(BaseComponent)+1]);
			int Grid_z = (LocGrid[(BaseComponent)+2]);

			//Trilinear Interpolation Distance Values
			double x_d = (LocVector[(BaseComponent)+0] * Inverse_Grid_Increment_Cuda) + X_center_offset - Grid_x;
			double y_d = (LocVector[(BaseComponent)+1] * Inverse_Grid_Increment_Cuda) + Y_center_offset - Grid_y;
			double z_d = (LocVector[(BaseComponent)+2] * Inverse_Grid_Increment_Cuda) + Z_center_offset - Grid_z;
			double nx_d = (1 - x_d);
			double ny_d = (1 - y_d);
			double nz_d = (1 - z_d);

			int BaseMap = (Grid_x * X_Field_Map) + (Grid_y * Y_Field_Map) + (Grid_z * 3);

			/* Cosine Trillinear Interpolation Distance Values
			double x_d = (1 - cos((((LocVector[(BaseComponent) + 0] + 500) - ((Grid_x) * Grid_Increment_Cuda)) / Grid_Increment_Cuda) * M_PI)) / 2;
			double y_d = (1 - cos((((LocVector[(BaseComponent) + 1] + 500) - ((Grid_y) * Grid_Increment_Cuda)) / Grid_Increment_Cuda) * M_PI)) / 2;
			double z_d = (1 - cos((((LocVector[(BaseComponent) + 2] + 500) - ((Grid_z) * Grid_Increment_Cuda)) / Grid_Increment_Cuda) * M_PI)) / 2;
			*/

			//Calculate Trillinear Interploated Values
			for (int i = 0; i < 3; i++) {
				double c_00 = (nx_d) * (Mag_Field[BaseMap + i]) + (x_d) * (Mag_Field[BaseMap + X_Field_Map + i]);
				double c_01 = (nx_d) * (Mag_Field[BaseMap + Z_Field_Map + i]) + (x_d) * (Mag_Field[BaseMap + X_Field_Map + Z_Field_Map + i]);
				double c_10 = (nx_d) * (Mag_Field[BaseMap + Y_Field_Map + i]) + (x_d) * (Mag_Field[BaseMap + X_Field_Map + Y_Field_Map + i]);
				double c_11 = (nx_d) * (Mag_Field[BaseMap + Y_Field_Map + Z_Field_Map + i]) + (x_d) * (Mag_Field[BaseMap + X_Field_Map + Y_Field_Map + Z_Field_Map + i]);
				double c_0 = (ny_d) * (c_00)+(c_10) * (y_d);
				double c_1 = (ny_d) * (c_01)+(c_11) * (y_d);
				double c = (nz_d) * (c_0)+(c_1) * (z_d);
				ResultField[(BaseComponent)+i] = c;
			}
		}
		else {
			for (int i = 0; i < 3; i++) {
				ResultField[(BaseComponent)+i] = 0;
			}
		}
	}
}

__global__ void TimeInterpolation(double* T1_Field, double* T2_Field, double *Result_Field, double CurrPoint) {
	int Comp = blockDim.x * blockIdx.x + threadIdx.x;
	if (Comp < TotFieldPoints) {
		double Time_Dist = CurrPoint * InterpInverse;
		double Value_Dist = T2_Field[Comp] - T1_Field[Comp];
		double Result_Value = ((1 - Time_Dist) * T1_Field[Comp]) + (Time_Dist * T2_Field[Comp]);
		Result_Field[Comp] = Result_Value;
	}
}

int main() {
	//Run Location and Velocity ini function
	Loc_ini();

	//Allocate some size stuff etc
	cudaError_t err = cudaSuccess;
	size_t size = 3 * TotCharge * sizeof(double);
	size_t FieldDataSize = TotFieldPoints * sizeof(double);

	//Open Input and Output Files
	ofstream myfile;
	myfile.open("example.txt");
	ofstream file;
	file.open("field.txt");
	ofstream tFile;
	tFile.open("Location.dat", ios::binary | ios::out);
	ofstream tVel;
	tVel.open("Velocity.dat", ios::binary | ios::out);
	ifstream tMag;
	ifstream tElec;
	ifstream fMag;
	ifstream fElec;

	//Allocate Memory in Cuda for Magnetic Field Grid Data and copy existing data to the CudaField
	double* CudaField = NULL;
	cudaMalloc((void**)&CudaField, (X_Dim * Y_Dim * Z_Dim * 3) * sizeof(double));
	double* CudaEField = NULL;
	cudaMalloc((void**)& CudaEField, (X_Dim * Y_Dim * Z_Dim * 3) * sizeof(double));

	//Allocate Memory in Cuda for Magnetic and Electric Field Grid Data imports
	double* CudaT1Field = NULL;
	cudaMalloc((void**)& CudaT1Field, (X_Dim * Y_Dim * Z_Dim * 3) * sizeof(double));
	double* CudaT2Field = NULL;
	cudaMalloc((void**)& CudaT2Field, (X_Dim * Y_Dim * Z_Dim * 3) * sizeof(double));
	double* CudaTE1Field = NULL;
	cudaMalloc((void**)& CudaTE1Field, (X_Dim * Y_Dim * Z_Dim * 3) * sizeof(double));
	double* CudaTE2Field = NULL;
	cudaMalloc((void**)& CudaTE2Field, (X_Dim * Y_Dim * Z_Dim * 3) * sizeof(double));

	//Allocate memory in Cuda for Location Arrays and Velocity Arrays
	double *CudaLoc = NULL;
	cudaMalloc((void **) &CudaLoc, size);
	double *CudaVel = NULL;
	cudaMalloc((void **) &CudaVel, size);

	//Allocate memory in Cuda for Particle Specific Magnetic Field Arrays and Electric Field Arrays
	double* CudaElec = NULL;
	cudaMalloc((void**)& CudaElec, size);
	double* CudaMag = NULL;
	cudaMalloc((void**)& CudaMag, size);

	//Allocate memory in Cuda for Particle Grid Reference array
	double* CudaGridLoc = NULL;
	cudaMalloc((void**)& CudaGridLoc, 3 * TotCharge * sizeof(double));

	cudaMemcpy(CudaLoc, Flat_Location_Vector, size, cudaMemcpyHostToDevice);
	cudaMemcpy(CudaVel, Flat_Velocity_Vector, size, cudaMemcpyHostToDevice);

	//Run Simulation for n Timesteps
	Count = 519;
	double Prog = 0;
	auto begin100 = std::chrono::steady_clock::now();
	auto end100 = std::chrono::steady_clock::now();
	auto begin500 = std::chrono::steady_clock::now();
	for (int n = 0; n < 10000; n++) {
		std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();

		//Write Flattened Location and Velocity Arrays into binary files every arbitrary number of steps.
		if (n % 10 == 0) {
			cudaMemcpy(Flat_Location_Vector, CudaLoc, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(Flat_Velocity_Vector, CudaVel, size, cudaMemcpyDeviceToHost);
			tFile.write((char*)Flat_Location_Vector, size);
			tVel.write((char*)Flat_Velocity_Vector, size);
		}

		//Print n every 100 steps
		if (n % 100 == 0) {
			end100 = std::chrono::steady_clock::now();
			std::cout << "Time difference (100 Frame Timing) = " << std::chrono::duration_cast<std::chrono::microseconds>(end100 - begin100).count() << "[µs]" << std::endl;
			cout << std::to_string(n) + "\r";
			begin100 = std::chrono::steady_clock::now();
		}

		//Step through field every N steps, The field files imported use the naming convention "Frame_Mag.dat" or "Frame_Elec.dat"
		if (n % InterpFrames == 0) {
			Prog = 0;
			Count++;
			if (Count == Steps) {
				Count = 0;
			}
			tMag.open("BinaryFieldData/" + std::to_string(Count) + "_Mag.dat", ios::binary | ios::in);
			tMag.read((char*)& T1_Field, FieldDataSize);
			tMag.close();
			fMag.open("BinaryFieldData/" + std::to_string(Count + 1) + "_Mag.dat", ios::binary | ios::in);
			fMag.read((char*)& T2_Field, FieldDataSize);
			fMag.close();
			cudaMemcpy(CudaT1Field, T1_Field, FieldDataSize, cudaMemcpyHostToDevice);
			cudaMemcpy(CudaT2Field, T2_Field, FieldDataSize, cudaMemcpyHostToDevice);
			tElec.open("BinaryFieldData/" + std::to_string(Count) + "_Elec.dat", ios::binary | ios::in);
			tElec.read((char*)& TE1_Field, FieldDataSize);
			tElec.close();
			fElec.open("BinaryFieldData/" + std::to_string(Count + 1) + "_Elec.dat", ios::binary | ios::in);
			fElec.read((char*)& TE2_Field, FieldDataSize);
			fElec.close();
			cudaMemcpy(CudaTE1Field, TE1_Field, FieldDataSize, cudaMemcpyHostToDevice);
			cudaMemcpy(CudaTE2Field, TE2_Field, FieldDataSize, cudaMemcpyHostToDevice);
		}
		TimeInterpolation<<<8192, 256>>>(CudaT1Field, CudaT2Field, CudaField, Prog);
		TimeInterpolation<<<8192, 256>>>(CudaTE1Field, CudaTE2Field, CudaEField, Prog);
		Prog++;

		cudaMemcpy(M_Field_Temp, CudaField, FieldDataSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(E_Field_Temp, CudaEField, FieldDataSize, cudaMemcpyDeviceToHost);

		//GridFiner will populate reference Cuda Particle Grid Reference array, Interpolation will use trillinear interpolation to find Particle Specific Magnetic Field Arrays. Data will be copied back to a Flat_M_Field array
		GridFinder<<<4096, 256>>>(CudaLoc, CudaGridLoc);
		cudaDeviceSynchronize();

		Interpolation<<<4096, 256>>>(CudaLoc, CudaField, CudaGridLoc, CudaMag);
		Interpolation<<<4096, 256>>>(CudaLoc, CudaEField, CudaGridLoc, CudaElec);
		cudaDeviceSynchronize();

		//Boris Method used to simulate particle motion
		BorisMethod<<<4096, 256>>>(CudaElec, CudaMag, CudaLoc, CudaVel);
		cudaDeviceSynchronize();

		std::chrono::steady_clock::time_point end5 = std::chrono::steady_clock::now();

	}
	auto end500 = std::chrono::steady_clock::now();
	std::cout << "Time difference (500 Frame Timing) = " << std::chrono::duration_cast<std::chrono::microseconds>(end500 - begin500).count() << "[µs]" << std::endl;
	//Close remaining Files
	myfile.close();
	file.close();
	tFile.close();
	tVel.close();
}