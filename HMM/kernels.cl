//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define A(i,j) A[i*N+j]
#define A1(i,j) A1[i*N+j]
#define A_used(i,j) A_used[i*N+j]
#define TAU(i,m) TAU[i*M+m]
#define TAU1(i,m) TAU1[i*M+m]
#define TAU_used(i,m) TAU_used[i*M+m]
#define MU(z,i,m) MU[((i)*M+m)*Z+z]
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

#define pi 3.1415926535897932384626433832795f


// ÏĞÎÒÀÁÓËÈĞÎÂÀÒÜ g ÇÀĞÀÍÅÅ!!!
__kernel void calc_g(
		int n,
		int N, int M, int Z, int T, int K,
		__global float *SIG1,
		__global float *Otr,
		__global float *MU1,
		__global float * g)
{
	size_t t = get_global_id(0);
	size_t k = get_global_id(1);
	size_t im = get_global_id(2);
	size_t i,m;// = im%N, m = im%M;   /// error!
	if (M > N){
			m = im % M; i = im / M;
	}else {
			i = im % N; m = im / N;
	}

	int z;
	float det=1.0f,res=0.0f;
	//float tmp1,tmp2;
	if (n==-1)
		n=0;
	for (z=0;z<Z;z++)
	{
		//tmp1 = SIG1(z,z,i,m,n);
		//det *= tmp1;
		//tmp2 = Otr(k,t,z)-MU1(z,i,m,n);
		//res += tmp2*tmp2/tmp1;
		det *= SIG1(z,z,i,m,n);
		res += ( Otr(k,t,z)-MU1(z,i,m,n))*( Otr(k,t,z)-MU1(z,i,m,n))/SIG1(z,z,i,m,n);
	}
	res *= -0.5f;
	res = exp(res)/sqrt(pown(2.0f*pi,Z)*det);
	
	if(!isfinite(res))
	{
		res=0.0f;
	}

	g(t,k,i,m,n) = res;
}

__kernel void calcB(int n, 
					int N, int M, int Z, int T, int K, 
					__global float * TAU, 
					__global float * B, __global float *SIG,
					__global float *Otr, __global float *MU,
					__global float *g
					)
{
	int m;
	size_t i = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);
	B(i,t,k)=0.0f;
	for(m=0; m<M; m++)
		B(i,t,k) += TAU(i,m)*g(t,k,i,m,n);
	if(!isfinite(B(i,t,k)))
		B(i,t,k) = 0.0f;
}

__kernel void k_2_1(
					int N, int K,
					__global float * c, 
					__global float * alf, __global float * bet,
					__global float * alf_t, __global float * bet_t
					)
{
	size_t i = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);
	alf(t,i,k)=0.0f;
	bet(t,i,k)=0.0f;
	alf_t(t,i,k)=0.0f;
	bet_t(t,i,k)=0.0f;
}

__kernel void k_2_2(int N, int K, int T1,
					__global float * alf_t, __global float * PI,
					__global float * B, __global float * bet
					)
{	int T = T1+1;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	alf_t(0,i,k) = PI[i] * B(i,0,k);
	bet(T1,i,k) = 1.0f;
}

__kernel void k_2_3_1(int N, int K, int t,
					__global float * alf, 
					__global float * alf_t
					)
{
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	float atsum=0.0f;
	for(j=0; j<N; j++)
		atsum += alf_t(t,j,k);
	alf(t,i,k)=alf_t(t,i,k)/atsum;
	if(!isfinite(alf(t,i,k)))
	{
		alf(t,i,k) = 1.0f/N;
	}
}

__kernel void k_2_3_2(int N, int K, int T, int t,
					__global float * alf, 
					__global float * alf_t,
					__global float * A,
					__global float * B
					)
{
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	float atsum = 0.0f;
	for(j=0; j<N; j++)
		atsum += alf(t,j,k)*A(j,i);
	alf_t(t+1,i,k) = B(i,t+1,k)*atsum;
}

__kernel void k_2_3_3(int N, int K,
					__global float * c,
					__global float * alf, 
					__global float * alf_t
					)
{
	int i;
	size_t t = get_global_id(0);
	size_t k = get_global_id(1);
	for(i=0; i<N; i++){
		c(t,k) = alf(t,i,k) / alf_t(t,i,k);
		if(isfinite(c(t,k)))
			break;
	}
	if(!isfinite(c(t,k))){
		c(t,k) = 10000000.0f;
	}
}

__kernel void k_2_4(int N, int K, int T1,
					__global float * alf,
					__global float * alf_t,
					__global float * c
					)
{
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	float atsum = 0.0f;
	for(j=0; j<N; j++)
		atsum += alf_t(T1,j,k);
	alf(T1,i,k) = alf_t(T1,i,k)/atsum;
	if(!isfinite(alf(T1,i,k)))					//
			alf(T1,i,k) = 1.0f/N;	
	for(j=0;j<N;j++){							// !!
		c(T1,k) = alf(T1,j,k)/alf_t(T1,j,k);
		if(isfinite(c(T1,k)))
			break;
	}
	if(!isfinite(c(T1,k)))
		c(T1,k)=1000.0f;		
}


__kernel void k_2_5_1( int t, int N, int K,
					  __global float * bet_t, 
					  __global float * c,
					  __global float * bet  
					 )
{
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	bet_t(t+1,i,k) = c(t+1,k) * bet(t+1,i,k);
}


__kernel void k_2_5_2(int t, int N, int K, int T,
					  __global float * bet,
					  __global float * A,
					  __global float * B,
					  __global float * bet_t, 
					  __global float * alf
					 )
{	
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	for(j=0;j<N;j++)
		bet(t,i,k) += A(i,j)*B(j,t+1,k)*bet_t(t+1,j,k);
	if(alf(t,i,k)==1.0f/N || fabs(alf(t,i,k)-1.0f) < 0.01f)
		bet(t,i,k)=1.0f;
}


__kernel void k_2_6(int n,
					int N, int M, int K, int Z, int T, 
					__global float * gam,
					__global float * alf,
					__global float * bet,
					__global float * TAU,
					__global float * SIG,
					__global float * Otr,
					__global float * MU,
					__global float * gamd,
					__global float *g
					)
{
	int m;
	size_t i = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);
	gam(t,i,k) = alf(t,i,k) * bet(t,i,k);
	float atsum = 0.0f;
	for(m=0; m<M; m++)
		atsum += TAU(i,m)*g(t,k,i,m,n);		//2 times g()
	for(m=0; m<M; m++){
		gamd(t,i,m,k)=TAU(i,m)*g(t,k,i,m,n)*gam(t,i,k)/atsum;	//again g()
		if(!isfinite(gamd(t,i,m,k)))
			gamd(t,i,m,k)=TAU(i,m)*gam(t,i,k);
	}
}

__kernel void k_2_7(int N, int K, int T,
					__global float * ksi,
					__global float * alf,
					__global float * A,
					__global float * B,
					__global float * bet
					)
{
	int j,i;
	size_t ii = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);		// Optimize !!!
	i = ii/N; j = ii%N;
	//for(j=0; j<N; j++)
		ksi(t,i,j,k)=alf(t,i,k)*A(i,j)*B(j,t+1,k)*bet(t+1,j,k);
}


__kernel void k_3_1_1(__global float * gam_sum)
{
	size_t i = get_global_id(0);
	gam_sum[i] = 0.0f;
}

__kernel void k_3_1_2(int M, __global float * gamd_sum)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	gamd_sum[i*M+m] = 0.0f;
}

__kernel void k_3_2_1(
					int t, int k,
					int N, int K,
					__global float * gam_sum,
					__global float * gam,
					__global int * flag
					)
{
	size_t i = get_global_id(0);
	gam_sum[i] += gam(t,i,k);	///???
	//if(!isfinite(gam_sum[i]))
	//	atomic_dec(flag);
}

__kernel void k_3_2_2(
					int t, int k,
					 int N, int M, int K,
					__global float * gam_sum,
					__global float * gamd_sum,
					__global float * gamd,
					__global int * flag
					)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	float ttt = gamd_sum[i*M+m] += gamd(t,i,m,k);
	if(isfinite(ttt))
		gamd_sum[i*M+m]+=gamd(t,i,m,k);
	//if (!isfinite(gamd_sum[i*M+m]))
	//	atomic_dec(flag);
}

__kernel void k_3_3(int N, int K, __global float * PI, __global float * gam)
{
	int k;
	size_t i = get_global_id(0);
	PI[i] = 0.0f;
	for(k=0;k<K;k++)
	{
		PI[i] += gam(0,i,k);
	}
	PI[i]/=K;
}

__kernel void k_3_4(
					int K, int N, int T1,
					__global float * A, __global float * ksi,
					__global float * gam_sum, __global float * c
					)
{
	int k,t;
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	float tmp2 = 0.0f;
	for(k=0; k<K; k++)
		for(t=0; t<T1; t++)
			tmp2 += ksi(t,i,j,k)*c(t+1,k);
	A(i,j) = tmp2 / gam_sum[i];
}

__kernel void k_3_5(int M, 
					__global float * TAU, __global float * gamd_sum,
					__global float * gam_sum
					)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	TAU(i,m) = gamd_sum[i*M+m]/gam_sum[i];
}

__kernel void k_3_6(
					int N, int M, int K, int Z, int T, 
					__global float * MU, __global float * gamd,
					__global float * Otr, __global float * gamd_sum
					)
{
	int k, t;
	size_t i = get_global_id(0);
	size_t z = get_global_id(1);
	size_t m = get_global_id(2);
	float ttt;
	MU(z,i,m)=0.0f;
	for(k=0; k<K; k++)
		for(t=0; t<T;t++)
		{
			ttt = MU(z,i,m) + gamd(t,i,m,k)*Otr(k,t,z);	// opt
			if(isfinite(ttt))
				MU(z,i,m) += gamd(t,i,m,k)*Otr(k,t,z);
		}
	MU(z,i,m) /= gamd_sum[i*M+m];
}

__kernel void k_3_7(
					int N, int M, int Z, int K, int T,
					__global float * SIG, __global float * gamd,
					__global float * gamd_sum,
					__global float * MU, __global float * Otr
					)
{
	size_t i = get_global_id(0);
	size_t z = get_global_id(1);
	size_t m = get_global_id(2);
	int k,t; float ttt, tmp3;
	size_t z1 = z / Z, z2 = z % Z;
	SIG(z1,z2,i,m) = 0.0f;
	for(k=0; k<K; k++)
		for(t=0; t<T; t++)
		{
			//for(z3=0; z3<Z; z3++)
				//tmp3[z3] = Otr(k,t,z3) - MU(z3,i,m);		//optimize!!
			tmp3 = (Otr(k,t,z1) - MU(z1,i,m)) * (Otr(k,t,z2) - MU(z2,i,m));	
			ttt = SIG(z1,z2,i,m) + gamd(t,i,m,k)*tmp3;
			if(isfinite(ttt))
				SIG(z1,z2,i,m) += gamd(t,i,m,k)*tmp3;
		}
	SIG(z1,z2,i,m) /= gamd_sum[i*M+m];
}

__kernel void k_3_8(__global float * PI1, __global float * PI)
{
	size_t i = get_global_id(0);
	PI1[i]=PI[i];
}

__kernel void k_3_9(int N, __global float * A1, __global float * A)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	A1(i,j)=A(i,j);
}

__kernel void k_3_10(int M,__global float * TAU1, __global float * TAU)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	TAU1(i,m) = TAU(i,m);
}

__kernel void k_3_11(int N, int M, int Z, int n,__global float * MU1, __global float * MU)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	size_t z = get_global_id(2);
	MU1(z,i,m,n) = MU(z,i,m);			// TODO: optimize? numinit - only one
}

__kernel void k_3_12(int N, int M, int Z, int n, __global float * SIG1, __global float * SIG)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	size_t z = get_global_id(2);
	size_t z1 = z / Z, z2 = z % Z;
	SIG1(z1,z2,i,m,n) = SIG(z1,z2,i,m);	
}