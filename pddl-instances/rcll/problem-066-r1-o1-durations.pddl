(define (problem rcll-production-066-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    o1 - order
    wp1 - workpiece
    cg1 cg2 cg3 cb1 cb2 cb3 - cap-carrier
    C-BS C-CS1 C-CS2 C-DS - mps
    CYAN - team-color
  )
   
  (:init (order-delivery-window-open o1) (at 169.058 (not (order-delivery-window-open o1))) (can-commit-for-ontime-delivery o1) (at 30.50 (not (can-commit-for-ontime-delivery o1)))
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
   (order-gate o1 GATE-1)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.858680)
   (= (path-length C-BS INPUT C-CS1 INPUT) 3.119279)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 4.636602)
   (= (path-length C-BS INPUT C-CS2 INPUT) 8.008403)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 8.765324)
   (= (path-length C-BS INPUT C-DS INPUT) 8.050949)
   (= (path-length C-BS INPUT C-DS OUTPUT) 6.274428)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.858680)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 1.437311)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 4.488734)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 6.326435)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 7.616261)
   (= (path-length C-BS OUTPUT C-DS INPUT) 7.770487)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 5.993966)
   (= (path-length C-CS1 INPUT C-BS INPUT) 3.119279)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 1.437311)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.891368)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 5.527383)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 6.817211)
   (= (path-length C-CS1 INPUT C-DS INPUT) 6.971436)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 5.194915)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 4.636601)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 4.488734)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.891368)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 5.837395)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 5.233957)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 4.519583)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 2.743063)
   (= (path-length C-CS2 INPUT C-BS INPUT) 8.008403)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 6.326435)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 5.527384)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 5.837396)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.559475)
   (= (path-length C-CS2 INPUT C-DS INPUT) 5.196856)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 4.977568)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 8.765324)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 7.616261)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 6.817211)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 5.233957)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.559475)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 2.853051)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 2.633762)
   (= (path-length C-DS INPUT C-BS INPUT) 8.050949)
   (= (path-length C-DS INPUT C-BS OUTPUT) 7.770486)
   (= (path-length C-DS INPUT C-CS1 INPUT) 6.971436)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 4.519583)
   (= (path-length C-DS INPUT C-CS2 INPUT) 5.196856)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 2.853051)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.070114)
   (= (path-length C-DS OUTPUT C-BS INPUT) 6.274429)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 5.993966)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 5.194915)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 2.743063)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 4.977567)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 2.633763)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.070115)
   (= (path-length START INPUT C-BS INPUT) 2.715758)
   (= (path-length START INPUT C-BS OUTPUT) 1.033789)
   (= (path-length START INPUT C-CS1 INPUT) 1.102618)
   (= (path-length START INPUT C-CS1 OUTPUT) 4.996504)
   (= (path-length START INPUT C-CS2 INPUT) 5.991742)
   (= (path-length START INPUT C-CS2 OUTPUT) 7.281569)
   (= (path-length START INPUT C-DS INPUT) 7.435794)
   (= (path-length START INPUT C-DS OUTPUT) 5.659273))

  (:goal (order-fulfilled o1))
)
