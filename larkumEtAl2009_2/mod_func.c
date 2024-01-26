#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _ampa_reg();
extern void _cad2_reg();
extern void _glutamate_reg();
extern void _h_reg();
extern void _h2_reg();
extern void _hh3_reg();
extern void _ih_reg();
extern void _it2_reg();
extern void _kap_reg();
extern void _kca_reg();
extern void _kdf_reg();
extern void _Kdr_reg();
extern void _kdr2_reg();
extern void _km_reg();
extern void _SlowCa_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," ampa.mod");
fprintf(stderr," cad2.mod");
fprintf(stderr," glutamate.mod");
fprintf(stderr," h.mod");
fprintf(stderr," h2.mod");
fprintf(stderr," hh3.mod");
fprintf(stderr," ih.mod");
fprintf(stderr," it2.mod");
fprintf(stderr," kap.mod");
fprintf(stderr," kca.mod");
fprintf(stderr," kdf.mod");
fprintf(stderr," Kdr.mod");
fprintf(stderr," kdr2.mod");
fprintf(stderr," km.mod");
fprintf(stderr," SlowCa.mod");
fprintf(stderr, "\n");
    }
_ampa_reg();
_cad2_reg();
_glutamate_reg();
_h_reg();
_h2_reg();
_hh3_reg();
_ih_reg();
_it2_reg();
_kap_reg();
_kca_reg();
_kdf_reg();
_Kdr_reg();
_kdr2_reg();
_km_reg();
_SlowCa_reg();
}
