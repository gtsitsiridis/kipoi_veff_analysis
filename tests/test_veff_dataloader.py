from kipoi_enformer.veff.dataloader import VCF_Enformer_DL

def test_dataloader():
    dl = VCF_Enformer_DL(
        fasta_file="example_files/fasta_file",
        gtf_file="example_files/gtf_file",
        vcf_file="example_files/promoter_variants.chr22.vcf",
    )
    for i in dl:
        print(i)

if __name__ == "__main__":
    test_dataloader()
