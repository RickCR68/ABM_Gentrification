# Agent Based Gentrification

## The Idea

Consists of making changes to the default Segretation model (included in mesa) and evaluating how these might impact the model's stability or time to stability, and maybe seeing if these introduce dynamics like genrifiaction. The main changes include:

1. Changing from binary 'blue/red' groups to a gradient of values, and having neighbourhoods be defined by likeness;
2. Introducing a neighbourhood "aproval" before an agent's move;
3. ...

## (Provisory) Work division:

## Analysis:

> [!NOTE]  
> modeling parameters and requirements not yet defined

### ~Parameters~ Initial Conditions:
- $N$: Size of grid;
- a

## Running:

### Web View (Solara)

```shell
# from root folder
solara run src/visualization/app.py
# if it doesn't work, try:
python -m solara run src/visualization/app.py
# Maybe windows is different, if so, update this readme plz
```
