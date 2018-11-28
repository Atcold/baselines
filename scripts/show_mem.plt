set term png small size 800,600
set output "mem-graph1.png"

set ylabel "VSZ"
set y2label "%MEM"

set ytics nomirror
set y2tics nomirror in

set yrange [0:*]
set y2range [0:*]

plot "./mem1.log" using 2 with lines axes x1y1 title "VSZ", \
     "./mem1.log" using 1 with lines axes x1y2 title "%MEM"

set output "mem-graph2.png"

set ylabel "VSZ"
set y2label "%MEM"

set ytics nomirror
set y2tics nomirror in

set yrange [0:*]
set y2range [0:*]

plot "./mem2.log" using 2 with lines axes x1y1 title "VSZ", \
     "./mem2.log" using 1 with lines axes x1y2 title "%MEM"

set output "mem-graph3.png"

set ylabel "VSZ"
set y2label "%MEM"

set ytics nomirror
set y2tics nomirror in

set yrange [0:*]
set y2range [0:*]

plot "./mem3.log" using 2 with lines axes x1y1 title "VSZ", \
     "./mem3.log" using 1 with lines axes x1y2 title "%MEM"
