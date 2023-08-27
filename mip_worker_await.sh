wpat="w[0-9]+"
while [[ "$(screen -ls)" =~ $wpat ]]; do
    sleep 5
done