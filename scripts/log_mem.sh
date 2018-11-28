while true; do
ps -p $1 -o %mem=,vsz= >> mem1.log
ps -p $2 -o %mem=,vsz= >> mem2.log
ps -p $3 -o %mem=,vsz= >> mem3.log
gnuplot show_mem.plt
sleep 1
done
