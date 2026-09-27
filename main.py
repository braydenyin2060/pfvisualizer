import matplotlib.pyplot as plt
import pandas as pd
import stkandttkcalculator as calculator
import mplcursors
from matplotlib.collections import LineCollection

def load_guns(csv_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Loads the gun stats CSV into a DataFrame. If your file is tab-delimited
    rather than comma-delimited, pass sep='\t' (or sep=None, engine='python'
    to auto-detect).
    """
    return pd.read_csv(csv_path, **read_csv_kwargs)


def get_gun(df: pd.DataFrame, gun_name: str) -> pd.Series:
    """
    Looks up a single gun's row by name (case-insensitive), so you don't
    have to write the df[df["gun_name"] == ...].iloc[0] boilerplate every
    time. Raises a clear error if the name doesn't match anything.
    """
    matches = df[df["gun_name"].str.lower() == gun_name.lower()]
    if matches.empty:
        raise ValueError(f"No gun named '{gun_name}' found in the dataframe.")
    return matches.iloc[0]
 
def damage_at_range(row: pd.Series, distance: float) -> float:
    """
    Base (unmultiplied) damage at a given distance, for one gun row.
    This is the building block for the later "categorize by STK across
    range" method — not needed for lowest_stk, but you'll want it next.
    """
    d0, r0, d1, r1 = row["damage0"], row["range0"], row["damage1"], row["range1"]
    if distance <= r0:
        return d0
    if distance >= r1:
        return d1
    t = (distance - r0) / (r1 - r0)
    return d0 + t * (d1 - d0)

guns = load_guns("data/guns.csv")

def damage_on_graph(gun, x):
    if x <= gun["range0"]:
        return gun["damage0"]

    elif x <= gun["range1"]:
        return gun["damage0"] + (
            (gun["damage1"] - gun["damage0"])
            * (x - gun["range0"])
            / (gun["range1"] - gun["range0"])
        )

    else:
        return gun["damage1"]

def plot_damage(gun_list):
    lines = []

    for gun in gun_list:
        line, = plt.plot(
            [0, gun["range0"], gun["range1"],300],
            [gun["damage0"], gun["damage0"], gun["damage1"], gun["damage1"]],
            label=gun["gun_name"],
            linewidth=5
        )

        lines.append((line, gun))
    
    gun1 = gun_list[0]
    gun2 = gun_list[1]

    for start in range(0, 300, 1):
        end = start + 1

        winner = calculator.compare_ttk(
            gun1,
            gun2,
            start
        )

        if winner is gun1:
            color = lines[0][0].get_color()

        elif winner is gun2:
            color = lines[1][0].get_color()

        else:
            color = "gray"

        for gun in gun_list:
            segment = [
                [
                    [start, damage_on_graph(gun, start)],
                    [end, damage_on_graph(gun, end)]
                ]
            ]

            collection = LineCollection(
                segment,
                colors=color,
                linewidths=4
            )

            plt.gca().add_collection(collection)

    plt.xlabel("Range (studs)")
    plt.ylabel("Damage")
    plt.title("Damage vs. Range")
    plt.ylim(0, 50)
    plt.legend()

    cursor = mplcursors.cursor(
        [line for line, gun in lines],
        hover=True
    )

    guide_line = plt.axvline(
        x=0,
        linewidth=1,
        linestyle="--",
        visible=False
    )

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target

        guide_line.set_xdata([x, x])
        guide_line.set_visible(True)

        text = f"Range: {x:.1f} studs\n\n"

        for line, gun in lines:
            damage = damage_at_range(gun, x)
            stk_ttk = calculator.stk_ttk_at_range(gun, x)

            text += f"{gun['gun_name']}\n"
            text += f"Damage: {damage:.1f}\n"

            for shots, ttk in stk_ttk:
                text += f"{shots}SK ({ttk:.3f} sec)\n"

            text += "\n"

        sel.annotation.set_text(text)

        plt.draw()


    def on_mouse_move(event):
        if event.inaxes != plt.gca():
            guide_line.set_visible(False)
            plt.draw()
            return

        for line, gun in lines:
            contains, _ = line.contains(event)

            if contains:
                guide_line.set_xdata([event.xdata, event.xdata])
                guide_line.set_visible(True)
                plt.draw()
                return

        guide_line.set_visible(False)
        plt.draw()


    plt.gcf().canvas.mpl_connect("motion_notify_event", on_mouse_move)

    for distance in range(0, 201, 10):
        winner = calculator.compare_ttk(gun_list[0], gun_list[1], distance)

        if winner is None:
            print(distance, "Tie")
        else:
            print(distance, winner["gun_name"])
    plt.show()
    

m16a3 = get_gun(guns, "MK16 SOCOM")
m38a7 = get_gun(guns, "M38A7")

plot_damage([m16a3, m38a7])
