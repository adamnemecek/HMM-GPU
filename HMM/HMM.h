#pragma once
#include <utility>
//#define __NO_STD_VECTOR // Use std::vector instead of STL version
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "svm.h"

#define MAX_ITER 5	// ������������ ����� �������� � ��������� �����-�����
//#define CONFIG_USE_DOUBLE             // ������������ �� ������� ��������

#if defined(CONFIG_USE_DOUBLE)
typedef cl_double real_t;
#else
typedef cl_float real_t;
#endif

#define A(i,j) A[(i)*N+j]
#define A1(i,j) A1[(i)*N+j]
#define A_used(i,j) A_used[(i)*N+j]
#define TAU(i,m) TAU[(i)*M+m]
#define TAU1(i,m) TAU1[(i)*M+m]
#define TAU_used(i,m) TAU_used[(i)*M+m]
#define MU(z,i,m) MU[((i)*M+m)*Z+z]
//#define MU(z,i,m,n) MU[(((n)*N+i)*M+m)*Z+z]
#define SIG(z1,z2,i,m) SIG[ (((i)*M+m)*Z+z1)*Z+ z2]
#define Otr(k,t,z) Otr[((k)*T+t)*Z+z]
#define MU1(z,i,m,n) MU1[(((n)*N+i)*M+m)*Z+z]
#define SIG1(z1,z2,i,m,n) SIG1[((((n)*N+i)*M+m)*Z+z1)*Z+z2]
#define B(i,t,k) B[((i)*T+t)*K+k]

#define c(t,k) c[(t)*K+k]
#define alf(t,i,k) alf[((t)*N+i)*K+k]
#define alf_t(t,i,k) alf_t[((t)*N+i)*K+k]
#define bet(t,i,k) bet[((t)*N+i)*K+k]
#define bet_t(t,i,k) bet_t[((t)*N+i)*K+k]
#define gam(t,i,k) gam[((t)*N+i)*K+k]
#define gamd(t,i,m,k) gamd[(((t)*N+i)*M+m)*K+k]
#define ksi(t,i,j,k) ksi[(((t)*N+i)*N+j)*K+k]

#define g(t,k,i,m,n) g[((((n)*N+i)*M+m)*K+k)*T+t]

#define pi 3.1415926535897932384626433832795

///
/// ��������� ��������������� svm
///
struct svm_scaling_parameters
{
	real_t lower;
	real_t upper;
	real_t * feature_min;
	real_t * feature_max;
	void clear()
	{
		delete feature_max;
		delete feature_min;
	}
};

class HMM
{
public:
	// ��������� ������
	cl_int N, M, K, T, NumInit, Z;
	real_t *PI,*A,*TAU,*MU,*SIG,*MU1,*SIG1,*Otr,*A1,*PI1,*TAU1;
	cl::Buffer *PI_b,*A_b,*TAU_b,*MU_b,*SIG_b,*MU1_b,*SIG1_b,*Otr_b,*A1_b,*PI1_b,*TAU1_b;
	real_t *c,*ksi,*gam,*gamd;
	cl::Buffer *alf_b,*bet_b,*c_b,*ksi_b,*gam_b,*gamd_b,*alf_t_b,*bet_t_b,*B_b;
	cl::Buffer * gam_sum_b, * gamd_sum_b;
	cl::Buffer * flag_b;
	cl::Buffer * g_b;
	cl::Buffer *cd_b, *alf_t_d_b, *alf_s_d_b, *alf1_PI_b, *alf1_MUSIG_b, *alf1_TAU_b,
		*a_A_b, *b_MUSIG_b, *b_TAU_b, *dets_b, *alf1_zero_b, *a_zero_b, *b_zero_b;
	cl::Buffer *d_PI_b, *d_A_b, *d_TAU_b, *d_MU_b, *d_SIG_b;
private: 
	cl::Context * context;							// ��������
	std::map<std::string,cl::Kernel*> kernels;		// ������ ��������
	cl::CommandQueue * queue;						// ��������� �������
public:
	// �������� ���������� ������ �� �����
	HMM(std::string filename);					
	~HMM(void);
	// ���������� ���������� ������
	void findModelParameters();					
	// ���������� �������� ������������������� ��� �������������
	void getObservations(std::string filename);
	// ���������� ������������������� ���������� � ������ Otr 
	void getObservations(std::string filename, real_t * Otr);		
	///
	/// ������������� ������������������� ���������� 
	///	@out: p[k] - ����������� ����, ��� ������ ������ �������� ������������������ � �������� k
	///
	void classifyObservations(real_t * p);		
	// �������� OpenCL ���������� � �������� �������
	void bindOpenCL(cl::Context * context, std::map<std::string,cl::Kernel*> & kernels, cl::CommandQueue * queue); 
	// ��������� ������ ��� ����������� �� ���������� ��� K �������������������
	void allocateDerivatives(int K);
	///
	/// �������� �� ������ ����������� (���� ��� ������ ��� ������)
	/// @in: observations - ����������, K - ����� �������������������, models - ������������� ������, numModels - ����� �������
	/// @out: scalingParams - ���������, �������������� ��� ��������������� ����������� ����� ��������� (�������� ������ ���������)
	/// @return: svmTrainedModel * - ��������� SVM ������		
	static svm_model * trainWithDerivatives(real_t ** observations, int K, HMM ** models, int numModels, svm_scaling_parameters & scalingParams);
	// ������ � ������� ����������� ��� ���������� �����
	void calcDerivatives(real_t * observations, int nOfSequences, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG);
private:
	// ���������� ���������� ����������� �����-�����
	real_t calcBaumWel�h(cl_int n);
	// ��������������� �������, ��� �������� � �������� � �������������
	void internal_calculations(cl_int n);
	// ���������� ����������� ��������� ������� ����������
	real_t calcProbability();

	// ������ ����������� ��� ���� �������������������
	void calc_derivatives_for_all_sequences();
};

void checkErr(cl_int err, const char * name);

