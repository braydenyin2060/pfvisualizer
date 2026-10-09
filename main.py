import matplotlib.pyplot as plt
import mplcursors
from matplotlib.collections import LineCollection
import data_loader
import stkandttkcalculator as calculator  # or: import stkandttkcalculator as calculator
 
X_MAX = 300  # furthest range (studs) to graph
 
 
def sample_curve(damage_ranges, gun_id):
    """Damage at every whole stud from 0..X_MAX, using the calculator's own math."""
    xs = list(range(X_MAX + 1))
    ys = [calculator.damage_at_range(damage_ranges, gun_id, x) for x in xs]
    return xs, ys
 
 
def plot_damage(data, gun_list):
    damage_ranges = data["damage_ranges"]
    ax = plt.gca()
 
    lines = []   # (matplotlib line, gun row)
    curves = {}  # gun_id -> list of damage values, index = stud
    for gun in gun_list:
        xs, ys = sample_curve(damage_ranges, gun["gun_id"])
        curves[gun["gun_id"]] = ys
        line, = ax.plot(xs, ys, label=gun["gun_name"], linewidth=5)
        lines.append((line, gun))
 
    gun1, gun2 = gun_list[0], gun_list[1]
    id1, id2 = gun1["gun_id"], gun2["gun_id"]
 
    # Group the 1-stud segments by (gun, color) so we build a few
    # LineCollections instead of hundreds.
    segments = {}
    for start in range(X_MAX):
        end = start + 1
        winner_id = calculator.compare_ttk(data, id1, id2, start)
        if winner_id == id1:
            color = lines[0][0].get_color()
        elif winner_id == id2:
            color = lines[1][0].get_color()
        else:
            color = "gray"
 
        for gun in gun_list:
            ys = curves[gun["gun_id"]]
            segments.setdefault((gun["gun_id"], color), []).append(
                [[start, ys[start]], [end, ys[end]]]
            )
 
    for (_, color), segs in segments.items():
        ax.add_collection(LineCollection(segs, colors=color, linewidths=4))
 
    ax.set_xlabel("Range (studs)")
    ax.set_ylabel("Damage")
    ax.set_title("Damage vs. Range")
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, max(max(ys) for ys in curves.values()) * 1.1)
    ax.legend()
 
    cursor = mplcursors.cursor([line for line, _ in lines], hover=True)
    guide_line = ax.axvline(x=0, linewidth=1, linestyle="--", visible=False)
 
    @cursor.connect("add")
    def on_add(sel):
        x, _ = sel.target
        guide_line.set_xdata([x, x])
        guide_line.set_visible(True)
 
        text = f"Range: {x:.1f} studs\n\n"
        for _, gun in lines:
            gid = gun["gun_id"]
            damage = calculator.damage_at_range(damage_ranges, gid, x)
            text += f"{gun['gun_name']}\nDamage: {damage:.1f}\n"
            for shots, ttk in calculator.stk_ttk_at_range(data, gid, x):
                text += f"{shots}SK ({ttk:.3f} sec)\n"
            text += "\n"
 
        sel.annotation.set_text(text)
        plt.gcf().canvas.draw_idle()
 
    def on_mouse_move(event):
        was_visible = guide_line.get_visible()
        hovering = False
        if event.inaxes == ax:
            hovering = any(line.contains(event)[0] for line, _ in lines)
            if hovering:
                guide_line.set_xdata([event.xdata, event.xdata])
        guide_line.set_visible(hovering)
        if hovering or was_visible:  # only redraw when something changed
            plt.gcf().canvas.draw_idle()
 
    plt.gcf().canvas.mpl_connect("motion_notify_event", on_mouse_move)
    plt.show()
 
 
def main():
    data = data_loader.load_all("data")
    guns = data["gun_names"]  # check data.keys() if this key name is different
 
    gun_a = calculator.get_gun(guns, "ak-47")
    gun_b = calculator.get_gun(guns, "F90 MBR")
 
    # Quick text readout of who wins at each distance
    for distance in range(0, 201, 10):
        winner_id = calculator.compare_ttk(data, gun_a["gun_id"], gun_b["gun_id"], distance)
        if winner_id is None:
            print(distance, "Tie")
        else:
            print(distance, gun_a["gun_name"] if winner_id == gun_a["gun_id"] else gun_b["gun_name"])
 
    plot_damage(data, [gun_a, gun_b])
 
 
if __name__ == "__main__":
    main()
 