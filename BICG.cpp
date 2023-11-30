//#ifndef __wtypes_h__
//#include <wtypes.h>
//#endif

//#ifndef __WINDEF_
//#include <windef.h>
//#endif
#ifdef _WIN32
// Windows Header Files
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include "nr3.h"
#include "sort.h"
#include "linbcg.h"
#include "sparse.h"
#include "asolve.h"


#ifdef _WIN32
#define EXPORTIT extern "C" __declspec( dllexport )

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
#else
#define EXPORTIT extern "C"
#endif
/*
BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpvReserved )  // reserved
{
    // Perform actions based on the reason for calling.
    switch( fdwReason ) 
    { 
        case DLL_PROCESS_ATTACH:
         // Initialize once for each new process.
         // Return FALSE to fail DLL load.
            break;

        case DLL_THREAD_ATTACH:
         // Do thread-specific initialization.
            break;

        case DLL_THREAD_DETACH:
         // Do thread-specific cleanup.
            break;

        case DLL_PROCESS_DETACH:
        
            if (lpvReserved != nullptr)
            {
                break; // do not do cleanup if process termination scenario
            }
            
         // Perform any necessary cleanup.
            break;
    }
    return TRUE;  // Successful DLL_PROCESS_ATTACH.
}*/

class coomat{
	public:
	int I, J;
	double V;
	coomat(int _I, int _J, double _V){
		I=_I;
		J=_J;
		V=_V;
	}
	bool operator <(const coomat & rhs){
		return this->J < rhs.J || (this->J==rhs.J && this->I < rhs.I);
	}
};

EXPORTIT void LinBCGSolve(int r, int c, int n, int * I, int * J, double * V, double *b, double *x, double tol = 1e-5, int maxiter=1000, int itol=1) {
	
	std::vector<coomat> cm;
	char str[1024];
	for (int i=0; i<n; i++){
		cm.push_back(coomat(I[i], J[i], V[i]));
	}
	std::sort(cm.begin(), cm.end());
	NRsparseMat A(r, c, n);
	A.col_ptr[cm[0].J] = 0;
	A.row_ind[0] = cm[0].I;
	A.val[0] = cm[0].V;
	for (int i=1; i<n; i++){
		if (cm[i].J != cm[i-1].J)	A.col_ptr[cm[i].J] = i;
		A.row_ind[i] = cm[i].I;
		A.val[i] = cm[i].V;
	}
	A.col_ptr[c] = n;
	NRsparseLinbcg bicg(A);
	VecDoub_I bo(r, b);
	VecDoub_IO xo(c, x);
	Int iter;
	Doub err;
	bicg.solve(bo, xo, itol, tol, maxiter, iter, err);
	sprintf(str, "BICG iterations: %d, Error: %f\n",iter,err);
	cout << str;
	for (int i=0; i<c; i++) x[i] = xo[i];
	
}

EXPORTIT void LinBCGSolveA(int r, int c, int n, int * indptr, int * indices, double * V, double *b, double *x, double tol = 1e-5, int maxiter=1000, int itol=1) {
	char str[1024];
	NRsparseMat A(r, c, n);
	for (int i=0; i<n; i++){
		A.row_ind[i] = indices[i];
		A.val[i] = V[i];
	}
	for (int i=0; i<=r; i++){
		A.col_ptr[i] = indptr[i];
	}
	NRsparseLinbcg bicg(A);
	VecDoub_I bo(r, b);
	VecDoub_IO xo(c, x);
	Int iter;
	Doub err;
	bicg.solve(bo, xo, itol, tol, maxiter, iter, err);
	sprintf(str, "BICG iterations: %d, Error: %f\n",iter,err);
	cout << str;
	for (int i=0; i<c; i++) x[i] = xo[i];
	
}
