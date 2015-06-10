//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#if CONFIG_USE_DOUBLE
	#if defined(cl_khr_fp64)  // Khronos extension available?
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		#define DOUBLE_SUPPORT_AVAILABLE
	#elif defined(cl_amd_fp64)  // AMD extension available?
		#pragma OPENCL EXTENSION cl_amd_fp64 : enable
		#define DOUBLE_SUPPORT_AVAILABLE
	#endif
#endif // CONFIG_USE_DOUBLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)
	// double
	typedef double real_t;
	typedef double2 real2_t;
	typedef double3 real3_t;
	typedef double4 real4_t;
	typedef double8 real8_t;
	typedef double16 real16_t;
	#define pi 3.14159265358979323846
#else
	// float
	typedef float real_t;
	typedef float2 real2_t;
	typedef float3 real3_t;
	typedef float4 real4_t;
	typedef float8 real8_t;
	typedef float16 real16_t;
	#define pi 3.14159265359f
#endif

#define A(i,j) A[(i)*N+j]
#define A1(i,j) A1[(i)*N+j]
#define A_used(i,j) A_used[(i)*N+j]
#define TAU(i,m) TAU[(i)*M+m]
#define TAU1(i,m) TAU1[(i)*M+m]
#define TAU_used(i,m) TAU_used[(i)*M+m]
#define MU(z,i,m) MU[((i)*M+m)*Z+z]
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

//#define pi 3.1415926535897932384626433832795f


// g табулируется заранее
// ()
__kernel void calc_g(
		int n,
		int N, int M, int Z, int T, int K,
		__global real_t *SIG1,
		__global real_t *Otr,
		__global real_t *MU1,
		__global real_t * g)
{
	size_t t = get_global_id(0);
	size_t k = get_global_id(1);
	size_t im = get_global_id(2);
	size_t i,m;// = im%N, m = im%M;		/// TODO: check
	if (M > N){
			m = im % M; i = im / M;
	}else {
			i = im % N; m = im / N;
	}

	int z;
	real_t det = 1.0f, res = 0.0f;		// TODO: whyyyy det=1???

	if (n==-1)
		n=0;
	for (z=0; z<Z; z++)
	{
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

// (49) - (2)
__kernel void calcB(int n, 
					int N, int M, int Z, int T, int K, 
					__global real_t * TAU, 
					__global real_t * B, __global real_t *SIG,
					__global real_t *Otr, __global real_t *MU,
					__global real_t *g
					)
{
	int m;
	size_t i = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);
	B(i,t,k)=0.0f;
	if (n == -1)
		n = 0;
	for(m=0; m<M; m++)
		B(i,t,k) += TAU(i,m)*g(t,k,i,m,n);
	if(!isfinite(B(i,t,k)))
		B(i,t,k) = 0.0f;
}

__kernel void k_2_1(
					int N, int K,
					__global real_t * c, 
					__global real_t * alf, __global real_t * bet,
					__global real_t * alf_t, __global real_t * bet_t
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
					__global real_t * alf_t, __global real_t * PI,
					__global real_t * B, __global real_t * bet
					)
{	int T = T1+1;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	alf_t(0,i,k) = PI[i] * B(i,0,k);
	bet(T1,i,k) = 1.0f;
}

__kernel void k_2_3_1(int N, int K, int t,
					__global real_t * alf, 
					__global real_t * alf_t
					)
{
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	real_t atsum=0.0f;
	for(j=0; j<N; j++)
		atsum += alf_t(t,j,k);
	alf(t,i,k)=alf_t(t,i,k)/atsum;
	if(!isfinite(alf(t,i,k)))
	{
		alf(t,i,k) = 1.0f/N;
	}
}

__kernel void k_2_3_2(int N, int K, int T, int t,
					__global real_t * alf, 
					__global real_t * alf_t,
					__global real_t * A,
					__global real_t * B
					)
{
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	real_t atsum = 0.0f;
	for(j=0; j<N; j++)
		atsum += alf(t,j,k)*A(j,i);
	alf_t(t+1,i,k) = B(i,t+1,k)*atsum;
	/*real_t temp = B(i,t+1,k)*atsum;
	if (isfinite(temp))
		alf_t(t+1,i,k) = temp;
	else
		alf_t(t+1,i,k) = 0;*/
}

__kernel void k_2_3_3(int N, int K,
					__global real_t * c,
					__global real_t * alf, 
					__global real_t * alf_t
					)
{
	int i;
	size_t t = get_global_id(0);
	size_t k = get_global_id(1);
	for(i=0; i<N; i++){
		c(t,k) = alf(t,i,k) / alf_t(t,i,k);		// TODO: strange
		if(isfinite(c(t,k)))
			break;
	}
	if(!isfinite(c(t,k))){
		c(t,k) = 10000000.0f;
	}
}

__kernel void k_2_4(int N, int K, int T1,
					__global real_t * alf,
					__global real_t * alf_t,
					__global real_t * c
					)
{
	int j;
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	real_t atsum = 0.0f;
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


// TODO: почему бы 2_5_1 и 2_5_2 не объединить?
__kernel void k_2_5_1( int t, int N, int K,
					  __global real_t * bet_t, 
					  __global real_t * c,
					  __global real_t * bet  
					 )
{
	size_t i = get_global_id(0);
	size_t k = get_global_id(1);
	bet_t(t+1,i,k) = c(t+1,k) * bet(t+1,i,k);
}


__kernel void k_2_5_2(int t, int N, int K, int T,
					  __global real_t * bet,
					  __global real_t * A,
					  __global real_t * B,
					  __global real_t * bet_t, 
					  __global real_t * alf
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
					__global real_t * gam,
					__global real_t * alf,
					__global real_t * bet,
					__global real_t * TAU,
					__global real_t * SIG,
					__global real_t * Otr,
					__global real_t * MU,
					__global real_t * gamd,
					__global real_t *g
					)
{
	int m;
	size_t i = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);
	gam(t,i,k) = alf(t,i,k) * bet(t,i,k);		// TODO: check, not corresponds with formula
	real_t atsum = 0.0f;
	for(m=0; m<M; m++)
		atsum += TAU(i,m)*g(t,k,i,m,n);			
	for(m=0; m<M; m++){							// TODO: can be parralelized
		gamd(t,i,m,k) = TAU(i,m) * g(t,k,i,m,n) * gam(t,i,k) / atsum;	
		if(!isfinite(gamd(t,i,m,k)))
			gamd(t,i,m,k)=TAU(i,m)*gam(t,i,k);
	}
}

__kernel void k_2_7(int N, int K, int T,
					__global real_t * ksi,
					__global real_t * alf,
					__global real_t * A,
					__global real_t * B,
					__global real_t * bet
					)
{
	int j,i;
	size_t ii = get_global_id(0);
	size_t t = get_global_id(1);
	size_t k = get_global_id(2);		
	i = ii/N; j = ii%N;
	ksi(t,i,j,k)=alf(t,i,k)*A(i,j)*B(j,t+1,k)*bet(t+1,j,k);
}


__kernel void k_3_1_1(__global real_t * gam_sum)
{
	size_t i = get_global_id(0);
	gam_sum[i] = 0.0f;
}

__kernel void k_3_1_2(int M, __global real_t * gamd_sum)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	gamd_sum[i*M+m] = 0.0f;
}

__kernel void k_3_2_1(
					int t, int k,
					int N, int K,
					__global real_t * gam_sum,
					__global real_t * gam,
					__global int * flag
					)
{
	size_t i = get_global_id(0);
	gam_sum[i] += gam(t, i, k);
}

__kernel void k_3_2_2(
					int t, int k,
					 int N, int M, int K,
					__global real_t * gam_sum,
					__global real_t * gamd_sum,
					__global real_t * gamd,
					__global int * flag
					)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	real_t ttt = gamd_sum[i*M+m] + gamd(t,i,m,k);
	if(isfinite(ttt))
		gamd_sum[i*M+m]+=gamd(t,i,m,k);
}

__kernel void k_3_3(int N, int K, __global real_t * PI, __global real_t * gam)
{
	int k;
	size_t i = get_global_id(0);
	PI[i] = 0.0f;
	for(k=0;k<K;k++)
	{
		PI[i] += gam(0,i,k);					// 1d-редукция
	}
	PI[i]/=K;
}

__kernel void k_3_4(
					int K, int N, int T1,
					__global real_t * A, __global real_t * ksi,
					__global real_t * gam_sum, __global real_t * c
					)
{
	int k,t;
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	real_t tmp2 = 0.0f;
	for(k=0; k<K; k++)							// 2d-редукция
		for(t=0; t<T1; t++)
			tmp2 += ksi(t,i,j,k)*c(t+1,k);
	A(i,j) = tmp2 / gam_sum[i];
}

__kernel void k_3_5(int M, 
					__global real_t * TAU, __global real_t * gamd_sum,
					__global real_t * gam_sum
					)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	TAU(i,m) = gamd_sum[i*M+m]/gam_sum[i];
}

__kernel void k_3_6(
					int N, int M, int K, int Z, int T, 
					__global real_t * MU, __global real_t * gamd,
					__global real_t * Otr, __global real_t * gamd_sum
					)
{
	int k, t;
	size_t i = get_global_id(0);
	size_t z = get_global_id(1);
	size_t m = get_global_id(2);
	real_t ttt;
	MU(z,i,m)=0.0f;
	for(k=0; k<K; k++)									// 2d-редукция
		for(t=0; t<T;t++)								
		{
			ttt = MU(z,i,m) + gamd(t,i,m,k)*Otr(k,t,z);
			if(isfinite(ttt))
				MU(z,i,m) += gamd(t,i,m,k)*Otr(k,t,z);
		}
	/*ttt = MU(z,i,m) / gamd_sum[i*M+m];
	if (isfinite(ttt))
		MU(z,i,m) = ttt;
	else
		MU(z,i,m) = 0.0f;*/
	MU(z,i,m) /= gamd_sum[i*M+m];
}

__kernel void k_3_7(
					int N, int M, int Z, int K, int T,
					__global real_t * SIG, __global real_t * gamd,
					__global real_t * gamd_sum,
					__global real_t * MU, __global real_t * Otr
					)
{
	size_t i = get_global_id(0);
	size_t z = get_global_id(1);
	size_t m = get_global_id(2);
	int k,t; real_t ttt, tmp3;
	size_t z1 = z / Z, z2 = z % Z;
	SIG(z1,z2,i,m) = 0.0f;
	for(k=0; k<K; k++)										// 2d-редукция
		for(t=0; t<T; t++)
		{
			tmp3 = (Otr(k,t,z1) - MU(z1,i,m)) * (Otr(k,t,z2) - MU(z2,i,m));	
			ttt = SIG(z1,z2,i,m) + gamd(t,i,m,k)*tmp3;
			if(isfinite(ttt))
				SIG(z1,z2,i,m) += gamd(t,i,m,k)*tmp3;
		}
	
	ttt = SIG(z1,z2,i,m) / gamd_sum[i*M+m];
	/*if (isfinite(ttt))
		SIG(z1,z2,i,m) = ttt;
	else
		SIG(z1,z2,i,m) = 0.0f;*/
	SIG(z1,z2,i,m) /= gamd_sum[i*M+m];
}

__kernel void k_3_8(__global real_t * PI1, __global real_t * PI)
{
	size_t i = get_global_id(0);
	PI1[i]=PI[i];
}

__kernel void k_3_9(int N, __global real_t * A1, __global real_t * A)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	A1(i,j)=A(i,j);
}

__kernel void k_3_10(int M,__global real_t * TAU1, __global real_t * TAU)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	TAU1(i,m) = TAU(i,m);
}

__kernel void k_3_11(int N, int M, int Z, int n,__global real_t * MU1, __global real_t * MU)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	size_t z = get_global_id(2);
	MU1(z,i,m,n) = MU(z,i,m);			// TODO: optimize? numinit - only one
}

__kernel void k_3_12(int N, int M, int Z, int n, __global real_t * SIG1, __global real_t * SIG)
{
	size_t i = get_global_id(0);
	size_t m = get_global_id(1);
	size_t z = get_global_id(2);
	size_t z1 = z / Z, z2 = z % Z;
	SIG1(z1,z2,i,m,n) = SIG(z1,z2,i,m);	
}



#define d_A(k,i,j) d_A[((k)*N+i)*N+j]
#define d_TAU(k,i,m) d_TAU[((k)*N+i)*M+m]
#define d_MU(k,z,i,m) d_MU[(((k)*N+i)*M+m)*Z+z]
#define d_SIG(k,z,i,m) d_SIG[(((k)*N+i)*M+m)*Z+z]
#define dets(k,i,m) dets[((k)*N+i)*M+m]
#define a_A(k,i,j,i1,j1) a_A[((((k)*N+i)*N+j)*N+i1)*N+j1]
#define b_MUSIG(k,i,m,z,i1,t) b_MUSIG[(((((k)*N+i)*M+m)*Z+z)*N+i1)*T+t]
#define b_TAU(k, i, m, i1, t) b_TAU[((((k)*N+i)*M+m)*N+i1)*T+t]
#define alf1_PI(k,i,j) alf1_PI[((k)*N+i)*N+j]
#define alf1_MUSIG(k, i, m, z, i1) alf1_MUSIG[((((k)*N+i)*M+m)*Z+z)*N+i1]
#define alf1_TAU(k, i, m, i1) alf1_TAU[(((k)*N+i)*M+m)*N+i1]

//#define cd(k,t) cd[(k)*T+t]
//#define alf_t_d(k, t, i) alf_t_d[((k)*T+t)*N+i]
//#define alf_s_d(k, t, i) alf_s_d[((k)*T+t)*N+i]

#define cd(k,i,j,m,z, t) cd[(((((k)*N+i)*N+j)*M+m)*Z+z)*T+t]
#define alf_t_d(k,i1,j,m,z, t,i) alf_t_d[(((((k)*N+i1)*N+j)*M+m)*Z+z)*N+i]
#define alf_s_d(k,i1,j,m,z, t,i) alf_s_d[(((((k)*N+i1)*N+j)*M+m)*Z+z)*N+i]

real_t calc_alpha_der(int k, int i1, int j1, int m1, int z1,
		int N, int M, int Z, int T, int K, 
		__global real_t * alf1_N, __global real_t * a_N, __global real_t * b_N, 
		__global real_t * cd, __global real_t * alf_t_d, __global real_t * alf_s_d,
		__global real_t * c, __global real_t * alf, __global real_t * alf_t, 
		__global real_t * A, __global real_t * B)
{
	int i, j, t;
	cd(k,i1,j1,m1,z1, 0) = 0;
	for (i = 0; i < N; i++)
	{
		alf_t_d(k,i1,j1,m1,z1, 0,i) = alf1_N[i];
		cd(k,i1,j1,m1,z1, 0) += alf_t_d(k,i1,j1,m1,z1, 0,i);
	}

	cd(k,i1,j1,m1,z1, 0) = -c(0, k) * c(0, k) * cd(k,i1,j1,m1,z1, 0);
	real_t sum1, sum2;
	for (t = 1; t<T; t++)
	{
		cd(k,i1,j1,m1,z1, t) = 0;
		for (i = 0; i<N; i++)
			alf_s_d(k,i1,j1,m1,z1, t-1,i) = 
				cd(k,i1,j1,m1,z1, t-1) * alf_t(t - 1, i, k) + alf_t_d(k,i1,j1,m1,z1, t-1, i) * c(t-1, k);
		for (i = 0; i<N; i++)
		{
			sum1 = sum2 = 0;
			for (j = 0; j<N; j++)
			{
				sum1 += alf_s_d(k,i1,j1,m1,z1, t-1,i) * A(j, i) + alf(t - 1, j, k) * a_N[j*N+i];
				sum2 += alf(t - 1, j, k) * A(j, i);
			}
			alf_t_d(k,i1,j1,m1,z1, t,i) = sum1 * B(i, t, k) + sum2 * b_N[i*T+t];
			cd(k,i1,j1,m1,z1, t) += alf_t_d(k,i1,j1,m1,z1, t,i);
		}
		cd(k,i1,j1,m1,z1, t) = -c(t, k) * c(t, k) * cd(k,i1,j1,m1,z1, t);
	}

	real_t deriv = 0;
	for (t = 0; t<T; t++)
		deriv += cd(k,i1,j1,m1,z1, t) / c(t, k);
	return deriv;
}

__kernel void k_4_0(__global real_t * arr)
{
	size_t i = get_global_id(0);
	arr[i] = 0.0;
}

__kernel void k_4_1_1(int N, int T, int K, __global real_t * alf1_PI, __global real_t * B)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t j = get_global_id(2);
	alf1_PI(k, i, j) = (j == i) ? B(i, 0, k) : 0;
}

__kernel void k_4_1_2(int N, int M, int Z, int T, int K, 
				__global real_t * alf1_PI, __global real_t * a_zero, __global real_t * b_zero,
				__global real_t * cd, __global real_t * alf_t_d, __global real_t * alf_s_d,
				__global real_t * c, __global real_t * alf, __global real_t * alf_t, 
				__global real_t * A, __global real_t * B, __global real_t * d_PI)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	real_t temp = calc_alpha_der(k,i,0,0,0,
									N, M, Z, T, K, &alf1_PI(k, i, 0), a_zero, b_zero, 
									cd, alf_t_d, alf_s_d, c, alf, alf_t, A, B);
	d_PI[k*N + i] = isfinite(temp) ? temp : 0.0;
}

__kernel void k_4_2_1(int N, __global real_t * a_A)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t j = get_global_id(2);
	int i1, j1;
	for (i1 = 0; i1 < N; i1++)
		for (j1 = 0; j1 < N; j1++)
			a_A(k, i, j, i1, j1) = (i1 == i && j1 == j) ? 1 : 0;
}

__kernel void k_4_2_2(int N, int M, int Z, int T, int K, 
				__global real_t * alf1_zero, __global real_t * a_A, __global real_t * b_zero,
				__global real_t * cd, __global real_t * alf_t_d, __global real_t * alf_s_d,
				__global real_t * c, __global real_t * alf, __global real_t * alf_t, 
				__global real_t * A, __global real_t * B, __global real_t * d_A)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t j = get_global_id(2);
	real_t temp = calc_alpha_der(k,i,j,0,0, 
									N, M, Z, T, K, alf1_zero, &a_A(k, i, j, 0, 0), b_zero,
									cd, alf_t_d, alf_s_d, c, alf, alf_t, A, B);
	d_A(k, i, j) = isfinite(temp) ? temp : 0.0;
}


__kernel void k_4_3_1(int N, int M, int Z, int T, int K,
				__global real_t * b_MUSIG, __global real_t * TAU, __global real_t * g,
				__global real_t * Otr, __global real_t * MU, __global real_t * SIG)
{
	size_t k = get_global_id(0);
	size_t t = get_global_id(1);
	size_t i = get_global_id(2);
	int m, z, i1;
	for (m = 0; m < M; m++)
		for (z = 0; z < Z; z++)
			for (i1 = 0; i1 < N; i1++)
				b_MUSIG(k, i, m, z, i1, t) = (i1 == i) ? 0.5 * TAU(i, m) * g(t, k, i, m, 0) * (Otr(k, t, z) - MU(z, i, m)) / SIG(z, z, i, m) : 0;
}

__kernel void k_4_3_2(int N, int M, int Z, int T,
				__global real_t * alf1_MUSIG, __global real_t * PI, __global real_t * b_MUSIG)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t m = get_global_id(2);
	int z, i1;
	for (z = 0; z < Z; z++)
		for (i1 = 0; i1 < N; i1++)
			alf1_MUSIG(k, i, m, z, i1) = PI[i] * b_MUSIG(k, i, m, z, i1, 0);
}

__kernel void k_4_3_3(int N, int M, int Z, int T, int K,
				__global real_t * alf1_MUSIG, __global real_t * a_zero, __global real_t * b_MUSIG,
				__global real_t * cd, __global real_t * alf_t_d, __global real_t * alf_s_d,
				__global real_t * c, __global real_t * alf, __global real_t * alf_t, 
				__global real_t * A, __global real_t * B, __global real_t * d_MU)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t mz = get_global_id(2);
	size_t m = mz / Z;
	size_t z = mz % Z;
	//for (z = 0; z < Z; z++)
	//{
		real_t temp = calc_alpha_der(k,i,0,m,z,
										N, M, Z, T, K, 
										&alf1_MUSIG(k, i, m, z, 0), a_zero, &b_MUSIG(k, i, m, z, 0, 0),
										cd, alf_t_d, alf_s_d, c, alf, alf_t, A, B);
		d_MU(k, z, i, m) = isfinite(temp) ? temp : 0.0;
	//}
}

__kernel void k_4_4_1(int N, int M, int Z,
				__global real_t * dets, __global real_t * SIG)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t m = get_global_id(2);
	int z;
	dets(k, i, m) = 1.0;
	for (z = 0; z < Z; z++)
		dets(k, i, m) *= SIG(z, z, i, m);
}

__kernel void k_4_4_2(int N, int M, int Z, int T, int K,
				__global real_t * b_MUSIG, __global real_t * TAU, __global real_t * g,
				__global real_t * Otr, __global real_t * MU, __global real_t * SIG,
				__global real_t * dets)
{
	size_t k = get_global_id(0);
	size_t t = get_global_id(1);
	size_t i = get_global_id(2);
	int m, z, i1;
	for (m = 0; m < M; m++)
		for (z = 0; z < Z; z++)
			for (i1 = 0; i1 < N; i1++)
				b_MUSIG(k, i, m, z, i1, t) = (i1 == i) ? TAU(i, m) * g(t, k, i, m, 0) * 0.5 * (pow((Otr(k, t, z) - MU(z, i, m)) / SIG(z, z, i, m), 2.0) - 1. / dets(k, i, m)) : 0;
				// optimize pow!!
}

__kernel void k_4_4_3(int N, int M, int Z, int T,
				__global real_t * alf1_MUSIG, __global real_t * PI, __global real_t * b_MUSIG)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t m = get_global_id(2);
	int z, i1;
	for (z = 0; z < Z; z++)
		for (i1 = 0; i1 < N; i1++)
			alf1_MUSIG(k, i, m, z, i1) = PI[i] * b_MUSIG(k, i, m, z, i1, 0);
}

__kernel void k_4_4_4(int N, int M, int Z, int T, int K,
				__global real_t * alf1_MUSIG, __global real_t * a_zero, __global real_t * b_MUSIG,
				__global real_t * cd, __global real_t * alf_t_d, __global real_t * alf_s_d,
				__global real_t * c, __global real_t * alf, __global real_t * alf_t, 
				__global real_t * A, __global real_t * B, __global real_t * d_SIG)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t mz = get_global_id(2);
	size_t m = mz / Z;
	size_t z = mz % Z;
	//for (z = 0; z < Z; z++)
	//{
		real_t temp = calc_alpha_der(k,i,0,m,z,
										N, M, Z, T, K,
										&alf1_MUSIG(k, i, m, z, 0), a_zero, &b_MUSIG(k, i, m, z, 0, 0),
										cd, alf_t_d, alf_s_d, c, alf, alf_t, A, B);
		d_SIG(k, z, i, m) = isfinite(temp) ? temp : 0.0;
	//}
}

__kernel void k_4_5_1(int N, int M, int Z, int T, int K,
				__global real_t * b_TAU, __global real_t * g)
{
	size_t k = get_global_id(0);
	size_t t = get_global_id(1);
	size_t i = get_global_id(2);
	int m, i1;
	for (m = 0; m < M; m++)
		for (i1 = 0; i1 < N; i1++)
			b_TAU(k, i, m, i1, t) = (i1 == i) ? g(t, k, i, m, 0) : 0;	// fixed!
}

__kernel void k_4_5_2(int N, int M, int Z, int T, int K,
				__global real_t * alf1_TAU, __global real_t * PI, __global real_t * b_TAU)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t m = get_global_id(2);
	int i1;
	for (i1 = 0; i1 < N; i1++)
		alf1_TAU(k, i, m, i1) = PI[i1] * b_TAU(k, i, m, i1, 0);			// fixed!
}

__kernel void k_4_5_3(int N, int M, int Z, int T, int K,
				__global real_t * alf1_TAU, __global real_t * a_zero, __global real_t * b_TAU,
				__global real_t * cd, __global real_t * alf_t_d, __global real_t * alf_s_d,
				__global real_t * c, __global real_t * alf, __global real_t * alf_t, 
				__global real_t * A, __global real_t * B, __global real_t * d_TAU)
{
	size_t k = get_global_id(0);
	size_t i = get_global_id(1);
	size_t m = get_global_id(2);
	real_t temp = calc_alpha_der(k,i,0,m,0,
									N, M, Z, T, K,
									&alf1_TAU(k, i, m, 0), a_zero, &b_TAU(k, i, m, 0, 0),
									cd, alf_t_d, alf_s_d, c, alf, alf_t, A, B);
	d_TAU(k, i, m) = isfinite(temp) ? temp : 0.0;
}