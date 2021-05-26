(define (problem rcll-production-072-durative)
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
   (order-base-color o1 BASE_BLACK)
   (order-cap-color o1 CAP_GREY)
   (order-gate o1 GATE-3)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.222936)
   (= (path-length C-BS INPUT C-CS1 INPUT) 5.808836)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 7.196317)
   (= (path-length C-BS INPUT C-CS2 INPUT) 10.634357)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 13.054346)
   (= (path-length C-BS INPUT C-DS INPUT) 7.171445)
   (= (path-length C-BS INPUT C-DS OUTPUT) 6.878916)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.222937)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 4.039282)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 6.827639)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 8.864805)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 11.284794)
   (= (path-length C-BS OUTPUT C-DS INPUT) 6.802767)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 6.510238)
   (= (path-length C-CS1 INPUT C-BS INPUT) 5.808836)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 4.039282)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.989519)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 7.076741)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 9.496729)
   (= (path-length C-CS1 INPUT C-DS INPUT) 4.964647)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 4.771167)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 7.196317)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 6.827639)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.989519)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 7.242470)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 6.880262)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 0.306371)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 3.100273)
   (= (path-length C-CS2 INPUT C-BS INPUT) 10.634358)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 8.864805)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 7.076741)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 7.242470)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.531412)
   (= (path-length C-CS2 INPUT C-DS INPUT) 7.336199)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 8.358357)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 13.054347)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 11.284794)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 9.496729)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 6.880260)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.531412)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 6.973989)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 7.996150)
   (= (path-length C-DS INPUT C-BS INPUT) 7.171445)
   (= (path-length C-DS INPUT C-BS OUTPUT) 6.802767)
   (= (path-length C-DS INPUT C-CS1 INPUT) 4.964647)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 0.306371)
   (= (path-length C-DS INPUT C-CS2 INPUT) 7.336199)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 6.973991)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.194002)
   (= (path-length C-DS OUTPUT C-BS INPUT) 6.878917)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 6.510238)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 4.771167)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 3.100273)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 8.358358)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 7.996150)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.194002)
   (= (path-length START INPUT C-BS INPUT) 2.680306)
   (= (path-length START INPUT C-BS OUTPUT) 0.762142)
   (= (path-length START INPUT C-CS1 INPUT) 4.070093)
   (= (path-length START INPUT C-CS1 OUTPUT) 6.858449)
   (= (path-length START INPUT C-CS2 INPUT) 8.307456)
   (= (path-length START INPUT C-CS2 OUTPUT) 10.727445)
   (= (path-length START INPUT C-DS INPUT) 6.833578)
   (= (path-length START INPUT C-DS OUTPUT) 6.541049))

  (:goal (order-fulfilled o1))
)
