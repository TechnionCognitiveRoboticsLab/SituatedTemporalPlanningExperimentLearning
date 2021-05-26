(define (problem rcll-production-081-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
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
   (order-gate o1 GATE-1)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.511727)
   (= (path-length C-BS INPUT C-CS1 INPUT) 3.669358)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 5.473462)
   (= (path-length C-BS INPUT C-CS2 INPUT) 8.512852)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 8.703949)
   (= (path-length C-BS INPUT C-DS INPUT) 8.480100)
   (= (path-length C-BS INPUT C-DS OUTPUT) 5.539652)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.511727)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 5.685791)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 7.711768)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 10.751157)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 10.062206)
   (= (path-length C-BS OUTPUT C-DS INPUT) 7.063454)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 5.903978)
   (= (path-length C-CS1 INPUT C-BS INPUT) 3.669358)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 5.685791)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.275041)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 7.314429)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 6.689962)
   (= (path-length C-CS1 INPUT C-DS INPUT) 5.472181)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 2.531734)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 5.473462)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 7.711768)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.275040)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 4.758006)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.949104)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 4.318625)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 4.497506)
   (= (path-length C-CS2 INPUT C-BS INPUT) 8.512850)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 10.751157)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 7.314429)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 4.758006)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.152833)
   (= (path-length C-CS2 INPUT C-DS INPUT) 7.377890)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 7.556770)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 8.703948)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 10.062207)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 6.689962)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.949104)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.152833)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 5.648035)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 5.826916)
   (= (path-length C-DS INPUT C-BS INPUT) 8.480099)
   (= (path-length C-DS INPUT C-BS OUTPUT) 7.063454)
   (= (path-length C-DS INPUT C-CS1 INPUT) 5.472181)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 4.318625)
   (= (path-length C-DS INPUT C-CS2 INPUT) 7.377890)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 5.648035)
   (= (path-length C-DS INPUT C-DS OUTPUT) 4.609135)
   (= (path-length C-DS OUTPUT C-BS INPUT) 5.539652)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 5.903978)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 2.531734)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 4.497506)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 7.556770)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 5.826916)
   (= (path-length C-DS OUTPUT C-DS INPUT) 4.609135)
   (= (path-length START INPUT C-BS INPUT) 1.118749)
   (= (path-length START INPUT C-BS OUTPUT) 3.357054)
   (= (path-length START INPUT C-CS1 INPUT) 3.248486)
   (= (path-length START INPUT C-CS1 OUTPUT) 4.411574)
   (= (path-length START INPUT C-CS2 INPUT) 7.450963)
   (= (path-length START INPUT C-CS2 OUTPUT) 7.642061)
   (= (path-length START INPUT C-DS INPUT) 7.972017)
   (= (path-length START INPUT C-DS OUTPUT) 5.118781))

  (:goal (order-fulfilled o1))
)
