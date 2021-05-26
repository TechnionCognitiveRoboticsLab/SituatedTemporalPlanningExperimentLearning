(define (problem rcll-production-061-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    R-2 - robot
    o1 - order
    wp1 - workpiece
    cg1 cg2 cg3 cb1 cb2 cb3 - cap-carrier
    C-BS C-CS1 C-CS2 C-DS - mps
    CYAN - team-color
  )
   
  (:init
   (mps-type C-BS BS)
   (mps-type C-CS1 CS)
   (mps-type C-CS2 CS) 
   (mps-type C-DS DS)
   (location-free START INPUT)
   (location-free C-BS INPUT)
   (location-free C-BS OUTPUT)
   (location-free C-CS1 INPUT)
   (location-free C-CS1 OUTPUT)
   (location-free C-CS2 INPUT)
   (location-free C-CS2 OUTPUT)
   (location-free C-DS INPUT)
   (location-free C-DS OUTPUT)
   (cs-can-perform C-CS1 CS_RETRIEVE)
   (cs-can-perform C-CS2 CS_RETRIEVE)
   (cs-free C-CS1)
   (cs-free C-CS2)

   (wp-base-color wp1 BASE_NONE)
   (wp-cap-color wp1 CAP_NONE)
   (wp-ring1-color wp1 RING_NONE)
   (wp-ring2-color wp1 RING_NONE)
   (wp-ring3-color wp1 RING_NONE)
   (wp-unused wp1)
   (robot-waiting R-1)
   (robot-waiting R-2)

   (mps-state C-BS IDLE)
   (mps-state C-CS1 IDLE)
   (mps-state C-CS2 IDLE)
   (mps-state C-DS IDLE)

   (wp-cap-color cg1 CAP_GREY)
   (wp-cap-color cg2 CAP_GREY)
   (wp-cap-color cg3 CAP_GREY)
   (wp-on-shelf cg1 C-CS1 LEFT)
   (wp-on-shelf cg2 C-CS1 MIDDLE)
   (wp-on-shelf cg3 C-CS1 RIGHT)

   (wp-cap-color cb1 CAP_BLACK)
   (wp-cap-color cb2 CAP_BLACK)
   (wp-cap-color cb3 CAP_BLACK)
   (wp-on-shelf cb1 C-CS2 LEFT)
   (wp-on-shelf cb2 C-CS2 MIDDLE)
   (wp-on-shelf cb3 C-CS2 RIGHT)
   (order-complexity o1 c0)
   (order-base-color o1 BASE_BLACK)
   (order-cap-color o1 CAP_GREY)
   (order-gate o1 GATE-2)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.178015)
   (= (path-length C-BS INPUT C-CS1 INPUT) 5.883924)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 5.480526)
   (= (path-length C-BS INPUT C-CS2 INPUT) 13.689311)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 9.419107)
   (= (path-length C-BS INPUT C-DS INPUT) 6.642715)
   (= (path-length C-BS INPUT C-DS OUTPUT) 8.705497)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.178015)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 4.213801)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 3.343000)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 11.889904)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 8.196562)
   (= (path-length C-BS OUTPUT C-DS INPUT) 5.420170)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 7.482952)
   (= (path-length C-CS1 INPUT C-BS INPUT) 5.883924)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 4.213801)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 2.828117)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 8.242567)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 5.541554)
   (= (path-length C-CS1 INPUT C-DS INPUT) 4.576121)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 4.972314)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 5.480527)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 3.343000)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 2.828116)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 9.002803)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 6.301791)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 5.503002)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.732551)
   (= (path-length C-CS2 INPUT C-BS INPUT) 13.689310)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 11.889903)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 8.242567)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 9.002804)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 5.565006)
   (= (path-length C-CS2 INPUT C-DS INPUT) 9.977172)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 6.845898)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 9.419107)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 8.196561)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 5.541555)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 6.301791)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 5.565006)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 5.575871)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 4.339268)
   (= (path-length C-DS INPUT C-BS INPUT) 6.642715)
   (= (path-length C-DS INPUT C-BS OUTPUT) 5.420169)
   (= (path-length C-DS INPUT C-CS1 INPUT) 4.576121)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 5.503001)
   (= (path-length C-DS INPUT C-CS2 INPUT) 9.977171)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 5.575871)
   (= (path-length C-DS INPUT C-DS OUTPUT) 4.108252)
   (= (path-length C-DS OUTPUT C-BS INPUT) 8.705499)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 7.482952)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 4.972314)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.732550)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 6.845898)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 4.339267)
   (= (path-length C-DS OUTPUT C-DS INPUT) 4.108253)
   (= (path-length START INPUT C-BS INPUT) 2.910338)
   (= (path-length START INPUT C-BS OUTPUT) 0.772811)
   (= (path-length START INPUT C-CS1 INPUT) 3.497394)
   (= (path-length START INPUT C-CS1 OUTPUT) 2.626592)
   (= (path-length START INPUT C-CS2 INPUT) 11.173495)
   (= (path-length START INPUT C-CS2 OUTPUT) 8.238227)
   (= (path-length START INPUT C-DS INPUT) 5.461834)
   (= (path-length START INPUT C-DS OUTPUT) 7.524617))

  (:goal (order-fulfilled o1))
)
