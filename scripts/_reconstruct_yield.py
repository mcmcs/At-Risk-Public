"""Reconstruct the manifest-declared yield CSV from the official NY Fed workbook.

This review helper reads no author data. Historical CSV layout is retained so
its bytes can be checked against the existing manifest, without weakening it.
Requires pandas and xlrd 2.0.2. The workbook is the public allmonth.xls linked
by https://www.newyorkfed.org/research/capital_markets/ycfaq.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json
import pandas as pd
import xlrd

def reconstruct(workbook, manifest, output):
    records=list(csv.DictReader(manifest.open()))
    record=next(r for r in records if r['input_id']=='yield_spread')
    first=pd.Timestamp(record['date_start']).to_period('M').to_timestamp()
    last=pd.Timestamp(record['date_end']).to_period('M').to_timestamp()
    book=xlrd.open_workbook(workbook)
    sheet=book.sheet_by_name('rec_prob')
    columns=sheet.row_values(0)
    expected=['Date','10 Year Treasury Yield','3 Month Treasury Yield','3 Month Treasury Yield (Bond Equivalent Basis)','Spread','Rec_prob','NBER_Rec']
    if columns!=expected: raise ValueError('Unexpected official workbook columns')
    rows=[]
    for i in range(1,sheet.nrows):
        row=sheet.row_values(i)
        raw=row[0]
        date=pd.to_datetime(raw,format='%d-%b-%Y') if isinstance(raw,str) else pd.Timestamp(xlrd.xldate.xldate_as_datetime(raw,book.datemode))
        date=date.to_period('M').to_timestamp()
        if first<=date<=last: rows.append([date,*row[1:]])
    if [x[0] for x in rows]!=list(pd.date_range(first,last,freq='MS')): raise ValueError('Missing or repeated sample month')
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\r\n')
    writer.writerow(columns)
    for row in rows:
        date=row[0]+pd.offsets.MonthEnd(0)
        # The historical CSV labels every February with day 28, even in leap years.
        if date.month==2: date=date.replace(day=28)
        date_string=date.strftime('%d-%b-%y') if date<=pd.Timestamp('2017-06-30') else f'{date.month}/{date.day}/{date.year}'
        values=[format(round(float(x),2),'g') for x in row[1:5]]
        probability='' if row[5]=='' else format(100*float(row[5]),'.2f')+'%'
        writer.writerow([date_string,*values,probability,str(int(row[6]))])
    content=stream.getvalue().removesuffix('\r\n').encode()
    digest=hashlib.sha256(content).hexdigest()
    if digest!=record['sha256'] or len(content)!=int(record['size_bytes']):
        raise ValueError(f'Official workbook no longer reconstructs the historical snapshot: {digest}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle: handle.write(content)
    return {'output':str(output),'rows':len(rows),'sha256':digest,'size_bytes':len(content),'manifest_match':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(reconstruct(args.workbook,args.manifest,args.output),indent=2))
