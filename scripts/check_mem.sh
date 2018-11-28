while true; do
ps -p $1 -o %mem=,vsz=
ps -p $2 -o %mem=,vsz=
ps -p $3 -o %mem=,vsz=
sleep 1
done
