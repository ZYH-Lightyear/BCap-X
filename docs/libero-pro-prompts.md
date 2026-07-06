# LIBERO-PRO Prompts Used by CaP-X Eval

CaP-X's LIBERO-PRO batch runner (`capx.envs.scripts.run_libero_batch`) defaults to
six 10-task suites:

- `libero_object_swap`
- `libero_object_task`
- `libero_goal_swap`
- `libero_goal_task`
- `libero_spatial_swap`
- `libero_spatial_task`

The prompt below is the LIBERO task language inserted into the CaP-X environment
prompt through `{libero_environment_goal}`. The `*_swap` and `*_task` variants share
the same task-language strings in the local LIBERO registry; their difference is in
the underlying LIBERO-PRO perturbation setup.

| Suite | Task ID | Task Name | Task Prompt |
|---|---:|---|---|
| `libero_object_swap` | 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | pick up the alphabet soup and place it in the basket |
| `libero_object_swap` | 1 | `pick_up_the_cream_cheese_and_place_it_in_the_basket` | pick up the cream cheese and place it in the basket |
| `libero_object_swap` | 2 | `pick_up_the_salad_dressing_and_place_it_in_the_basket` | pick up the salad dressing and place it in the basket |
| `libero_object_swap` | 3 | `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | pick up the bbq sauce and place it in the basket |
| `libero_object_swap` | 4 | `pick_up_the_ketchup_and_place_it_in_the_basket` | pick up the ketchup and place it in the basket |
| `libero_object_swap` | 5 | `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | pick up the tomato sauce and place it in the basket |
| `libero_object_swap` | 6 | `pick_up_the_butter_and_place_it_in_the_basket` | pick up the butter and place it in the basket |
| `libero_object_swap` | 7 | `pick_up_the_milk_and_place_it_in_the_basket` | pick up the milk and place it in the basket |
| `libero_object_swap` | 8 | `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | pick up the chocolate pudding and place it in the basket |
| `libero_object_swap` | 9 | `pick_up_the_orange_juice_and_place_it_in_the_basket` | pick up the orange juice and place it in the basket |
| `libero_object_task` | 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | pick up the alphabet soup and place it in the basket |
| `libero_object_task` | 1 | `pick_up_the_cream_cheese_and_place_it_in_the_basket` | pick up the cream cheese and place it in the basket |
| `libero_object_task` | 2 | `pick_up_the_salad_dressing_and_place_it_in_the_basket` | pick up the salad dressing and place it in the basket |
| `libero_object_task` | 3 | `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | pick up the bbq sauce and place it in the basket |
| `libero_object_task` | 4 | `pick_up_the_ketchup_and_place_it_in_the_basket` | pick up the ketchup and place it in the basket |
| `libero_object_task` | 5 | `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | pick up the tomato sauce and place it in the basket |
| `libero_object_task` | 6 | `pick_up_the_butter_and_place_it_in_the_basket` | pick up the butter and place it in the basket |
| `libero_object_task` | 7 | `pick_up_the_milk_and_place_it_in_the_basket` | pick up the milk and place it in the basket |
| `libero_object_task` | 8 | `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | pick up the chocolate pudding and place it in the basket |
| `libero_object_task` | 9 | `pick_up_the_orange_juice_and_place_it_in_the_basket` | pick up the orange juice and place it in the basket |
| `libero_goal_swap` | 0 | `open_the_middle_drawer_of_the_cabinet` | open the middle drawer of the cabinet |
| `libero_goal_swap` | 1 | `put_the_bowl_on_the_stove` | put the bowl on the stove |
| `libero_goal_swap` | 2 | `put_the_wine_bottle_on_top_of_the_cabinet` | put the wine bottle on top of the cabinet |
| `libero_goal_swap` | 3 | `open_the_top_drawer_and_put_the_bowl_inside` | open the top drawer and put the bowl inside |
| `libero_goal_swap` | 4 | `put_the_bowl_on_top_of_the_cabinet` | put the bowl on top of the cabinet |
| `libero_goal_swap` | 5 | `push_the_plate_to_the_front_of_the_stove` | push the plate to the front of the stove |
| `libero_goal_swap` | 6 | `put_the_cream_cheese_in_the_bowl` | put the cream cheese in the bowl |
| `libero_goal_swap` | 7 | `turn_on_the_stove` | turn on the stove |
| `libero_goal_swap` | 8 | `put_the_bowl_on_the_plate` | put the bowl on the plate |
| `libero_goal_swap` | 9 | `put_the_wine_bottle_on_the_rack` | put the wine bottle on the rack |
| `libero_goal_task` | 0 | `open_the_middle_drawer_of_the_cabinet` | open the middle drawer of the cabinet |
| `libero_goal_task` | 1 | `put_the_bowl_on_the_stove` | put the bowl on the stove |
| `libero_goal_task` | 2 | `put_the_wine_bottle_on_top_of_the_cabinet` | put the wine bottle on top of the cabinet |
| `libero_goal_task` | 3 | `open_the_top_drawer_and_put_the_bowl_inside` | open the top drawer and put the bowl inside |
| `libero_goal_task` | 4 | `put_the_bowl_on_top_of_the_cabinet` | put the bowl on top of the cabinet |
| `libero_goal_task` | 5 | `push_the_plate_to_the_front_of_the_stove` | push the plate to the front of the stove |
| `libero_goal_task` | 6 | `put_the_cream_cheese_in_the_bowl` | put the cream cheese in the bowl |
| `libero_goal_task` | 7 | `turn_on_the_stove` | turn on the stove |
| `libero_goal_task` | 8 | `put_the_bowl_on_the_plate` | put the bowl on the plate |
| `libero_goal_task` | 9 | `put_the_wine_bottle_on_the_rack` | put the wine bottle on the rack |
| `libero_spatial_swap` | 0 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | pick up the black bowl between the plate and the ramekin and place it on the plate |
| `libero_spatial_swap` | 1 | `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | pick up the black bowl next to the ramekin and place it on the plate |
| `libero_spatial_swap` | 2 | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | pick up the black bowl from table center and place it on the plate |
| `libero_spatial_swap` | 3 | `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | pick up the black bowl on the cookie box and place it on the plate |
| `libero_spatial_swap` | 4 | `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| `libero_spatial_swap` | 5 | `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | pick up the black bowl on the ramekin and place it on the plate |
| `libero_spatial_swap` | 6 | `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | pick up the black bowl next to the cookie box and place it on the plate |
| `libero_spatial_swap` | 7 | `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | pick up the black bowl on the stove and place it on the plate |
| `libero_spatial_swap` | 8 | `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | pick up the black bowl next to the plate and place it on the plate |
| `libero_spatial_swap` | 9 | `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | pick up the black bowl on the wooden cabinet and place it on the plate |
| `libero_spatial_task` | 0 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | pick up the black bowl between the plate and the ramekin and place it on the plate |
| `libero_spatial_task` | 1 | `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | pick up the black bowl next to the ramekin and place it on the plate |
| `libero_spatial_task` | 2 | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | pick up the black bowl from table center and place it on the plate |
| `libero_spatial_task` | 3 | `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | pick up the black bowl on the cookie box and place it on the plate |
| `libero_spatial_task` | 4 | `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| `libero_spatial_task` | 5 | `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | pick up the black bowl on the ramekin and place it on the plate |
| `libero_spatial_task` | 6 | `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | pick up the black bowl next to the cookie box and place it on the plate |
| `libero_spatial_task` | 7 | `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | pick up the black bowl on the stove and place it on the plate |
| `libero_spatial_task` | 8 | `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | pick up the black bowl next to the plate and place it on the plate |
| `libero_spatial_task` | 9 | `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | pick up the black bowl on the wooden cabinet and place it on the plate |

To regenerate this table from the local LIBERO registry:

```bash
source .venv-libero/bin/activate
python - <<'PY'
from libero import benchmark

suites = [
    "libero_object_swap",
    "libero_object_task",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_spatial_swap",
    "libero_spatial_task",
]

bd = benchmark.get_benchmark_dict()
for suite_name in suites:
    suite = bd[suite_name]()
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        prompt = getattr(task, "language", None) or getattr(task, "name", "")
        print(f"{suite_name}\t{i}\t{task.name}\t{prompt}")
PY
```
