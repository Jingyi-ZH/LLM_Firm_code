# Topic: Optimal design for conjoint study
# Project: LLM_Firm

## PlayStation 5 Pro

library(AlgDesign)

# Levels from config/apps/playstation5pro.yaml
launch_msrp_usd = c(499, 599, 699, 799)
cpu_cores = c(4, 8, 12)
gpu_peak_fp32_tflops = c(8, 12, 16, 20, 24)
system_memory_gb = c(12, 16, 18, 22)
internal_storage_gb = c(1000, 1500, 2000, 2500, 3000)
storage_type = c("HDD", "SSD")
max_output_vertical_resolution = c("2K", "4K", "8K")

Vs = list(
  launch_msrp_usd, cpu_cores, gpu_peak_fp32_tflops,
  system_memory_gb, internal_storage_gb, storage_type, max_output_vertical_resolution
)

# How many features
K = length(Vs)

# Combinations
levs = sapply(Vs, length)
prod(levs)
# Parameters
sum(levs - 1)

# Full Factorial Design
var_names = c(
  "launch_msrp_usd", "cpu_cores", "gpu_peak_fp32_tflops",
  "system_memory_gb", "internal_storage_gb", "storage_type", "max_output_vertical_resolution"
)

mat = gen.factorial(levels = levs, varNames = var_names, factors = "all")

head(mat)
tail(mat)

# Let's Generate data from this design
# Convert to a model matrix
mm1 = model.matrix(~., data = mat)
head(mm1)

# Can we find set of messages to test?
set.seed(12345)
frm = as.formula(paste0("~", paste0(var_names, collapse = "+")))
ds2 = optFederov(frm, data = mat, nTrials = 100, criterion = "I")
des = ds2$design

# Decode index levels (1..k) to actual attribute values.
level_map = list(
  launch_msrp_usd = launch_msrp_usd,
  cpu_cores = cpu_cores,
  gpu_peak_fp32_tflops = gpu_peak_fp32_tflops,
  system_memory_gb = system_memory_gb,
  internal_storage_gb = internal_storage_gb,
  storage_type = storage_type,
  max_output_vertical_resolution = max_output_vertical_resolution
)

des_decoded = des
for (v in var_names) {
  idx = as.integer(des[[v]])
  des_decoded[[v]] = level_map[[v]][idx]
}

# Keep the leading numeric order ID and add "PS" prefix.
des_decoded = cbind(
  real_profile_id = paste0("PS", seq_len(nrow(des_decoded))),
  des_decoded
)

# Export decoded design (directly usable levels).
write.csv(file = "./design_ps5pro.csv", des_decoded, row.names = FALSE)
