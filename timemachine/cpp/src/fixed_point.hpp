// #define FIXED_EXPONENT 0x20000000000
// #define FIXED_EXPONENT 0x100000000  // 2^32
#define FIXED_EXPONENT             0x100000000  // 2^36 
#define FIXED_BORN_PSI             0x40000000000 // 2^42
#define FIXED_EXPONENT_BORN_FORCES 0x100000000  // 2^36

// #define FIXED_EXPONENT             0x2000000000  // 2^36 
// #define FIXED_BORN_PSI             0x40000000000 // 2^42
// #define FIXED_EXPONENT_BORN_FORCES 0x2000000000  // 2^36 

// (ytz) it's difficult to extend this any bigger
// because we compute nb exclusions in a separate
// pass from the nonbonded calculation itself
