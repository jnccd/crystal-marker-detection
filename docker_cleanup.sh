docker kill $(docker ps -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)
docker system prune
docker system prune -af

# Admin powershell
#Optimize-VHD -Path "C:\Users\nikla\AppData\Local\Docker\wsl\data\ext4.vhdx" -Mode full
#Optimize-VHD -Path "C:\Users\nikla\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx" -Mode full