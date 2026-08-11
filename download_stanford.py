import redivis

# Télécharge area_1_no_xyz.tar directement sur le serveur
table = redivis.table("sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s:f304:v1_0.no_xyz:ct1f")
file = table.file("area_1_no_xyz.tar")
file.download("/home/william/data/stanford/area_1_no_xyz.tar")
print("Téléchargement terminé ✓")