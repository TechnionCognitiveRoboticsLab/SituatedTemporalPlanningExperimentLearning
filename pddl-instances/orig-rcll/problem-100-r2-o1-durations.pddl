(define (problem rcll-production-100-durative)
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
   (order-base-color o1 BASE_RED)
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-3)



   (= (path-length C-BS INPUT C-BS OUTPUT) 3.292624)
   (= (path-length C-BS INPUT C-CS1 INPUT) 6.934318)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 6.019304)
   (= (path-length C-BS INPUT C-CS2 INPUT) 10.309305)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 11.214233)
   (= (path-length C-BS INPUT C-DS INPUT) 7.441005)
   (= (path-length C-BS INPUT C-DS OUTPUT) 6.059499)
   (= (path-length C-BS OUTPUT C-BS INPUT) 3.292624)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 6.857922)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 7.450954)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 10.232909)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 11.137837)
   (= (path-length C-BS OUTPUT C-DS INPUT) 8.872655)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 7.491149)
   (= (path-length C-CS1 INPUT C-BS INPUT) 6.934317)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 6.857921)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.425193)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 5.202541)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 5.986071)
   (= (path-length C-CS1 INPUT C-DS INPUT) 3.913745)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 3.404712)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 6.019304)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 7.450953)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.425193)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 4.716618)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 5.500148)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 3.174807)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 0.324084)
   (= (path-length C-CS2 INPUT C-BS INPUT) 10.309305)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 10.232909)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 5.202541)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 4.716619)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.358591)
   (= (path-length C-CS2 INPUT C-DS INPUT) 5.205170)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 4.696138)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 11.214232)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 11.137836)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 5.986071)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 5.500148)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.358591)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 4.306540)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.479667)
   (= (path-length C-DS INPUT C-BS INPUT) 7.441006)
   (= (path-length C-DS INPUT C-BS OUTPUT) 8.872655)
   (= (path-length C-DS INPUT C-CS1 INPUT) 3.913745)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 3.174807)
   (= (path-length C-DS INPUT C-CS2 INPUT) 5.205170)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 4.306540)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.190521)
   (= (path-length C-DS OUTPUT C-BS INPUT) 6.059499)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 7.491148)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 3.404713)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 0.324084)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 4.696137)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.479667)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.190521)
   (= (path-length START INPUT C-BS INPUT) 2.131193)
   (= (path-length START INPUT C-BS OUTPUT) 2.054798)
   (= (path-length START INPUT C-CS1 INPUT) 5.555407)
   (= (path-length START INPUT C-CS1 OUTPUT) 6.148439)
   (= (path-length START INPUT C-CS2 INPUT) 8.930394)
   (= (path-length START INPUT C-CS2 OUTPUT) 9.835322)
   (= (path-length START INPUT C-DS INPUT) 7.570140)
   (= (path-length START INPUT C-DS OUTPUT) 6.188634))

  (:goal (order-fulfilled o1))
)
