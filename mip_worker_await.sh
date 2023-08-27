wpat="w[0-9]+"
if [[ "$(screen -ls)" =~ $wpat ]]; then
    sleep 5