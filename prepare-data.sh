unzip /mnt/vinai/sidd.zip || true
cd SIDD_Medium_Srgb/
mkdir val_data
mv Data/0199_010_GP_00800_01600_5500_N val_data/
mv Data/0198_010_GP_00100_00200_5500_N/ val_data/
mv Data/0197_009_IP_00100_00200_5500_L/ val_data/
mv Data/0196_009_IP_00800_02000_5500_L/ val_data/
mv Data/0200_010_GP_01600_03200_5500_N/ val_data/
mv val_data/0196_009_IP_00800_02000_5500_L/0196_NOISY_SRGB_010.PNG val_data/0196_009_IP_00800_02000_5500_L/0196_NOISY_SRGB_010.PNG.tmp
