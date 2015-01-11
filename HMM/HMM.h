#pragma once
#include <utility>
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>
#include <iostream>
#include <map>


#define A(i,j) A[i*N+j]
#define A1(i,j) A1[i*N+j]
#define A_used(i,j) A_used[i*N+j]
#define TAU(i,m) TAU[i*M+m]
#define TAU1(i,m) TAU1[i*M+m]
#define TAU_used(i,m) TAU_used[i*M+m]
#define MU(z,i,m) MU[((i)*M+m)*Z+z]
//#define MU(z,i,m,n) MU[(((n)*N+i)*M+m)*Z+z]
#define SIG(z1,z2,i,m) SIG[ ((i*M+m)*Z+z1)*Z+ z2]
#define Otr(k,t,z) Otr[(k*T+t)*Z+z]
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

class HMM
{
public:
	// ��������� ������
	cl_int N,M,K,T,NumInit;
	cl_int Z;	// ����������� ����������
	cl_float *PI,*A,*TAU,*MU,*SIG,*MU1,*SIG1,*Otr,*A1,*PI1,*TAU1;
	cl::Buffer *PI_b,*A_b,*TAU_b,*MU_b,*SIG_b,*MU1_b,*SIG1_b,*Otr_b,*A1_b,*PI1_b,*TAU1_b;
	cl_float *alf,*bet,*c,*ksi,*gam,*gamd,*alf_t,*bet_t,*B;
	cl::Buffer *alf_b,*bet_b,*c_b,*ksi_b,*gam_b,*gamd_b,*alf_t_b,*bet_t_b,*B_b;
	cl::Buffer * gam_sum_b, * gamd_sum_b;
	cl::Buffer * flag_b;
	cl::Buffer * g_b;
private: 
	cl::Context * context;							// ��������
	std::map<std::string,cl::Kernel*> kernels;		// ������ ��������
	cl::CommandQueue * queue;						// ��������� �������
public:
	HMM(std::string filenamMe);					// �������� ���������� ������ �� �����
	~HMM(void);
	void findModelParameters();					// ���������� ���������� ������
	void getTestObserv(std::string filename);	// ���������� �������� ������������������� ��� �������������
	void classifyObservations(cl_float * p);	// ������������� ������������������� ���������� 
												// (p[k] - ������������ ����, ��� 
												// ������ ������ �������� ������������������ ��� ������� k)
	void bindOpenCL(cl::Context * context_, std::map<std::string,cl::Kernel*> & kernels, cl::CommandQueue * queue_); // �������� OpenCL ���������� � �������� �������
private:
	// ���������� ���������� ����������� �����-�����
	cl_float calcBaumWel�h(cl_int n);
	// ��������������� �������, ��� �������� � �������� � �������������
	void internal_calculations(cl_int n);
	// ��������������� ������� ��� ��������
	//cl_float g(int t,int k,int i,int m,int n);
	// ���������� ����������� ��������� ������� ����������
	cl_float calcProbability();
	// �������� ������ (�����)
	//static void checkErr(cl_int err, const char * name);
};

void checkErr(cl_int err, const char * name);

