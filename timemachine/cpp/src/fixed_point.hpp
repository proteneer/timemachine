// #define FIXED_EXPONENT 0x20000000000
// #define FIXED_EXPONENT 0x100000000  // 2^32
#define FIXED_EXPONENT 0x1000000000 // 2^36 
#define FIXED_EXPONENT_BORN_FORCES 0x1000000000 // 2^36 
// (ytz) it's difficult to extend this any bigger
// because we compute nb exclusions in a separate
// pass from the nonbonded calculation itself
