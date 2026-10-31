// Forward R's routine registration to Rust, so the linker keeps the static library.
void R_init_dmlrs_extendr(void *dll);

void R_init_dmlrs(void *dll) {
    R_init_dmlrs_extendr(dll);
}
