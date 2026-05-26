# Topic: Optimal design for conjoint study
# Project: LLM_Firm

## iPhone 17e + base storage

library(AlgDesign)

# Levels from iPhone17e_basestorage/iphone17e_basestorage.yaml
base_storage = c("128 GB", "256 GB")
battery_life = c(18, 24, 30, 36)
screen_size = c(6.1, 6.3, 6.6, 6.9)
thickness = c(5.8, 6.8, 7.8, 8.3, 8.8)
front_camera = c(12, 18, 24, 30)
rear_camera = c(36, 48, 60)
focal_length = c(1, 2, 3, 5)
ultrawide = c("equipped", "not equipped")
geekbench = c(7000, 7800, 8600, 9400)
ram = c(4, 8, 12, 16)
price = c(499, 599, 699, 799, 899)

Vs = list(
  base_storage, battery_life, screen_size, thickness, front_camera,
  rear_camera, focal_length, ultrawide, geekbench, ram, price
)

# How many features
K = length(Vs)

# Combinations
levs = sapply(Vs, length)
prod(levs)
# Parameters
sum(levs - 1)

var_names = c(
  "base_storage", "battery_life", "screen_size", "thickness",
  "front_camera", "rear_camera", "focal_length", "ultrawide",
  "geekbench", "ram", "price"
)

# Full factorial design
mat = gen.factorial(levels = levs, varNames = var_names, factors = "all")

head(mat)
tail(mat)

# Convert to a model matrix
mm1 = model.matrix(~ ., data = mat)
head(mm1)

# Generate an optimal subset
set.seed(12345)
frm = as.formula(paste0("~", paste0(var_names, collapse = "+")))
ds2 = optFederov(frm, data = mat, nTrials = 100, criterion = "I")
des = ds2$design
des_decoded = des
for (v in var_names) {
  idx = as.integer(des[[v]])
  des_decoded[[v]] = level_map[[v]][idx]
}

# Keep the leading numeric order ID and add "E" prefix.
des_decoded = cbind(
  profile_id = paste0("E", seq_len(nrow(des_decoded))),
  des_decoded
)

write.csv(file = "./design_17e_basestorage.csv", des, row.names = FALSE)
