# Topic: Optimal design for conjoint study
# Project: LLM_Firm

## iPad mini 7

library(AlgDesign)

# Levels from config/apps/ipadmini7.yaml
launch_msrp_usd = c(399, 499, 599)
base_storage_gb = c(64, 128, 256)
cpu_cores = c(4, 6, 8, 10)
gpu_cores = c(4, 5, 6, 8, 10)
ram_size_gb = c(4, 6, 8, 12)
screen_size_inch = c(7.9, 8.3, 8.7)
bandwidth_gbps = c(80, 120, 160)

Vs = list(
  launch_msrp_usd, base_storage_gb, cpu_cores, gpu_cores,
  ram_size_gb, screen_size_inch, bandwidth_gbps
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
  "launch_msrp_usd", "base_storage_gb", "cpu_cores", "gpu_cores",
  "ram_size_gb", "screen_size_inch", "bandwidth_gbps"
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
ds2 = optFederov(frm, data = mat, nTrials = 120, criterion = "I")
des = ds2$design

# Decode index levels (1..k) to actual attribute values.
level_map = list(
  launch_msrp_usd = launch_msrp_usd,
  base_storage_gb = base_storage_gb,
  cpu_cores = cpu_cores,
  gpu_cores = gpu_cores,
  ram_size_gb = ram_size_gb,
  screen_size_inch = screen_size_inch,
  bandwidth_gbps = bandwidth_gbps
)

des_decoded = des
for (v in var_names) {
  idx = as.integer(des[[v]])
  des_decoded[[v]] = level_map[[v]][idx]
}

# Keep the leading numeric order ID and add "IM" prefix.
des_decoded = cbind(
  profile_id = paste0("IM", seq_len(nrow(des_decoded))),
  des_decoded
)

# Export decoded design (directly usable levels).
write.csv(file = "./design_ipadmini7.csv", des_decoded, row.names = FALSE)
