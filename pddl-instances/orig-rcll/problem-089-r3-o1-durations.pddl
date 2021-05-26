(define (problem rcll-production-089-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    R-2 - robot
    R-3 - robot
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
   (robot-waiting R-3)

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
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-1)



   (= (path-length C-BS INPUT C-BS OUTPUT) 3.013911)
   (= (path-length C-BS INPUT C-CS1 INPUT) 5.711157)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 5.944603)
   (= (path-length C-BS INPUT C-CS2 INPUT) 9.451179)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 11.771480)
   (= (path-length C-BS INPUT C-DS INPUT) 6.259044)
   (= (path-length C-BS INPUT C-DS OUTPUT) 5.960961)
   (= (path-length C-BS OUTPUT C-BS INPUT) 3.013911)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 6.224175)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 6.934992)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 7.214176)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 9.741386)
   (= (path-length C-BS OUTPUT C-DS INPUT) 7.249433)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 6.951350)
   (= (path-length C-CS1 INPUT C-BS INPUT) 5.711157)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 6.224174)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.593185)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 4.814120)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 7.134420)
   (= (path-length C-CS1 INPUT C-DS INPUT) 3.976054)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 5.965816)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 5.944603)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 6.934992)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.593185)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.337485)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.418832)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 0.709929)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 3.746543)
   (= (path-length C-CS2 INPUT C-BS INPUT) 9.451179)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 7.214177)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 4.814121)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.337485)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.548598)
   (= (path-length C-CS2 INPUT C-DS INPUT) 5.720354)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 7.710115)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 11.771480)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 9.741386)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 7.134421)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.418833)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.548599)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 6.801702)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 8.448579)
   (= (path-length C-DS INPUT C-BS INPUT) 6.259043)
   (= (path-length C-DS INPUT C-BS OUTPUT) 7.249432)
   (= (path-length C-DS INPUT C-CS1 INPUT) 3.976054)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 0.709929)
   (= (path-length C-DS INPUT C-CS2 INPUT) 5.720354)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 6.801702)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.129412)
   (= (path-length C-DS OUTPUT C-BS INPUT) 5.960962)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 6.951351)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 5.965816)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 3.746543)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 7.710115)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 8.448581)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.129412)
   (= (path-length START INPUT C-BS INPUT) 4.383549)
   (= (path-length START INPUT C-BS OUTPUT) 1.810831)
   (= (path-length START INPUT C-CS1 INPUT) 4.476021)
   (= (path-length START INPUT C-CS1 OUTPUT) 5.909182)
   (= (path-length START INPUT C-CS2 INPUT) 5.466023)
   (= (path-length START INPUT C-CS2 OUTPUT) 7.993232)
   (= (path-length START INPUT C-DS INPUT) 6.223622)
   (= (path-length START INPUT C-DS OUTPUT) 5.925539))

  (:goal (order-fulfilled o1))
)
