(define (problem rcll-production-048-durative)
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
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-1)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.178015)
   (= (path-length C-BS INPUT C-CS1 INPUT) 5.762557)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 5.968581)
   (= (path-length C-BS INPUT C-CS2 INPUT) 11.445461)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 10.416111)
   (= (path-length C-BS INPUT C-DS INPUT) 7.004344)
   (= (path-length C-BS INPUT C-DS OUTPUT) 8.932665)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.178015)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 3.923459)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 4.129482)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 9.606362)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 8.577012)
   (= (path-length C-BS OUTPUT C-DS INPUT) 5.999922)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 7.093566)
   (= (path-length C-CS1 INPUT C-BS INPUT) 5.762557)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 3.923458)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.361714)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 6.780199)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 5.750850)
   (= (path-length C-CS1 INPUT C-DS INPUT) 4.886717)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 4.267403)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 5.968580)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 4.129482)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.361714)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.837155)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 6.822912)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 6.916606)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.339465)
   (= (path-length C-CS2 INPUT C-BS INPUT) 11.445460)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 9.606362)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 6.780199)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.837155)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.370318)
   (= (path-length C-CS2 INPUT C-DS INPUT) 8.407138)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 6.484552)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 10.416111)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 8.577012)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 5.750850)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 6.822911)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.370318)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 5.987379)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 3.191433)
   (= (path-length C-DS INPUT C-BS INPUT) 7.004342)
   (= (path-length C-DS INPUT C-BS OUTPUT) 5.999921)
   (= (path-length C-DS INPUT C-CS1 INPUT) 4.886717)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 6.916605)
   (= (path-length C-DS INPUT C-CS2 INPUT) 8.407138)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 5.987380)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.312174)
   (= (path-length C-DS OUTPUT C-BS INPUT) 8.932665)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 7.093566)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 4.267404)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.339465)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 6.484552)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 3.191434)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.312174)
   (= (path-length START INPUT C-BS INPUT) 3.242250)
   (= (path-length START INPUT C-BS OUTPUT) 1.104723)
   (= (path-length START INPUT C-CS1 INPUT) 3.734931)
   (= (path-length START INPUT C-CS1 OUTPUT) 3.940954)
   (= (path-length START INPUT C-CS2 INPUT) 9.417834)
   (= (path-length START INPUT C-CS2 OUTPUT) 8.388485)
   (= (path-length START INPUT C-DS INPUT) 5.811394)
   (= (path-length START INPUT C-DS OUTPUT) 6.905038))

  (:goal (order-fulfilled o1))
)
