import pandas as pd
import os
import time
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from django.http import HttpResponse, JsonResponse
from ...app.ms1.utilities import connect_to_mongoDB, connect_to_mongo_gridfs
from pymongo.errors import OperationFailure
from gridfs.errors import NoFile
import io
import matplotlib as plt
from openpyxl.styles import Font, Border, Side, Alignment

# used for parallelization
# import concurrent.futures
# import collections

# os.environ['IN_DOCKER'] = "False" #for local dev - also see similar switch in app/nta_task.py


def datetime_handler(x):
    if isinstance(x, datetime):
        return x.isoformat()
    raise TypeError("Unknown type")


class OutputServer:
    """
    This class connects to mongodb and servers files from a give nta_run based on its jobID. It returns the files
    in the form of an HttpResponse (content type of either csv data or zip data), ready to be served up by the API.
    """

    def __init__(self, jobid="00000000", project_name=None):
        self.jobid = jobid
        self.project_name = ""
        self.in_docker = os.environ.get("IN_DOCKER") != "False"
        self.mongo_address = os.environ.get("MONGO_SERVER")
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.posts = self.mongo.posts

    def status(self):
        try:
            search_id = self.jobid + "_status"
            db_record = self.posts.find_one({"_id": search_id})
            status = db_record["status"]
            time = db_record["date"]
            except_text = db_record["error_info"]
        except TypeError:
            status = "Not found"
            time = "Not found"
            except_text = "Not found"
        response_data = {
            "start_time": time,
            "status": status,
            "error_info": except_text,
        }
        return JsonResponse(response_data)

    # -----------------------------------------------------------------
    #
    # BEGIN PARALLELIZATION OF XLSX GENERATION
    #
    # -----------------------------------------------------------------

    # def generate_excel_sheet(self, name):
    #     try:
    #         print(f'Constructing {name} file')
    #         start = time.perf_counter()
    #         id = self.jobid + "_" + name
    #         db_record = self.gridfs.get(id)
    #         json_string = db_record.read().decode('utf-8')
    #         df = pd.read_json(json_string, orient='split')
    #         return name, df
    #     except Exception as e:
    #         print(e)

    # def generate_excel(self):
    #     file_names = self.gridfs.get(f'{self.jobid}_file_names').read().decode('utf-8').split("&&")
    #     print(f'file_names: {file_names}')

    #     in_memory_buffer = BytesIO()
    #     sheet_dict = collections.OrderedDict()

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #         futures = [executor.submit(self.generate_excel_sheet, name) for name in file_names if 'final_' not in name]

    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 name, df = future.result()
    #                 sheet_dict[name] = df
    #             except Exception as e:
    #                 print(e)

    #     with pd.ExcelWriter(in_memory_buffer, engine='openpyxl') as writer:
    #         # for name, df in sheet_dict.items():
    #         #     print(f'name of sheet {name}')
    #         #     df.to_excel(writer, sheet_name=name, index=False)
    #         for name in file_names:
    #             if 'final_' not in name:
    #                 df = sheet_dict.get(name)
    #                 if df is not None:
    #                     print(f'name of sheet {name}')
    #                     df.to_excel(writer, sheet_name=name, index=False)

    #     return in_memory_buffer.getvalue()

    # -----------------------------------------------------------------
    #
    # END PARALLELIZATION OF XLSX GENERATION
    #
    # -----------------------------------------------------------------

    # def construct_excel_sheet(self, writer, name, file_names):
    #     try:
    #         print(f"Constructing {name} file")
    #         start = time.perf_counter()
    #         id = self.jobid + "_" + name
    #         db_record = self.gridfs.get(id)
    #         json_string = db_record.read().decode("utf-8")
    #         df = pd.read_json(json_string, orient="split")
    #         df.to_excel(writer, sheet_name=name, index=False)
    #         if name == "Chemical Results":
    #             sheet_number = file_names.index(name)
    #             workbook = writer.book
    #             # Access the Chemical Results sheet
    #             sheet = workbook.worksheets[sheet_number]
    #             # Set the style of the DTSXID column to hyperlink. The excel column number of the DTSXID column is always 8 as of 06Feb2025
    #             for i in range(sheet.max_row):
    #                 cell = sheet.cell(row=i + 2, column=8)
    #                 cell.style = "Hyperlink"

    #         # Adjust column widths of sheet - NTAW-470 AC 6/26/2024
    #         # Get max value of string length for entire column, add one to it, and set the column width to this value
    #         for column in df:
    #             column_width = max(df[column].astype(str).map(len).max(), len(column)) + 1
    #             col_idx = df.columns.get_loc(column)
    #             writer.sheets[name].set_column(col_idx, col_idx, column_width)

    #         stop = time.perf_counter()
    #         print(f"Time to construct {name}: {stop - start}")
    #     except Exception as e:
    #         print(e)

    # def generate_excel(self):
    #     file_names = self.gridfs.get(f"{self.jobid}_file_names").read().decode("utf-8").split("&&")
    #     print(f"file_names: {file_names}")
    #     in_memory_buffer = BytesIO()
    #     with pd.ExcelWriter(in_memory_buffer, engine="openpyxl") as writer:
    #         for name in file_names:
    #             if "final_" in name:
    #                 continue
    #             self.construct_excel_sheet(writer, name, file_names)
    #     return in_memory_buffer.getvalue()

    def add_tracer_plots_to_zip(self, zipf, jobid):
        tracer_plots = self.gridfs.get(f"{self.jobid}_tracer_files").read().decode("utf-8").split("&&")

        # #5/21/2024 AC: Save figures into buffer before zippping
        # for name in tracer_plots:
        #     try:
        #         tracer_id = jobid + "_" + name
        #         db_record = self.gridfs.get(tracer_id)

        #         # Grab the figure from storage
        #         buffer = db_record.read()

        #         # Save the figure into a buffer
        #         buf = io.BytesIO()
        #         buffer.savefig(buf, bbox_inches="tight", format="png")
        #         plt.close(buffer)
        #         buf.seek(0)

        #         project_name = db_record.project_name
        #         if project_name:
        #             filename = project_name.replace(" ", "_") + "_" + name + ".png"
        #         else:
        #             filename = tracer_id + ".png"
        #         zipf.writestr(filename, buf.read())
        #         # zipf.writestr(filename, buffer)
        #     except (OperationFailure, TypeError, NoFile) as e:
        #         break
        # 5/20/2024 AC: Comment out to debug tracer plots
        for name in tracer_plots:
            try:
                tracer_id = jobid + "_" + name
                db_record = self.gridfs.get(tracer_id)
                buffer = db_record.read()
                project_name = db_record.project_name
                if project_name:
                    filename = project_name.replace(" ", "_") + "_" + name + ".png"
                else:
                    filename = tracer_id + ".png"
                zipf.writestr(filename, buffer)
            except (OperationFailure, TypeError, NoFile) as e:
                break

    def add_cv_scatterplot_to_zip(self, zipf, jobid):
        try:
            id = jobid + "_cv_scatterplot"
            db_record = self.gridfs.get(id)
            buffer = db_record.read()
            project_name = db_record.project_name
            if project_name:
                filename = project_name.replace(" ", "_") + "_cv_scatterplot.png"
            else:
                filename = id + ".png"
            zipf.writestr(filename, buffer)
        except (OperationFailure, TypeError, NoFile) as e:
            pass

    def add_occurrence_heatmap_to_zip(self, zipf, jobid):
        # heatmap_plot = self.gridfs.get(f'{self.jobid}_occurrence_heatmaps').read().decode('utf-8').split("&&")
        try:
            heatmap_id = jobid + "_occurrence_heatmap"
            db_record = self.gridfs.get(heatmap_id)
            buffer = db_record.read()
            project_name = db_record.project_name
            if project_name:
                filename = project_name.replace(" ", "_") + "_occurrence_heatmap.png"
            else:
                filename = heatmap_id + ".png"
            zipf.writestr(filename, buffer)
        except (OperationFailure, TypeError, NoFile) as e:
            pass

    def final_result(self):
        in_memory_zip = BytesIO()

        with ZipFile(in_memory_zip, "w", ZIP_DEFLATED) as zipf:
            # excel_data = self.generate_excel()
            excel_data = self.gridfs.get(f"{self.jobid}_excel").read()

            project_name = str(self.gridfs.get(f"{self.jobid}_project_name").read(), "utf-8")
            filename = project_name.replace(" ", "_") + "_NTA_WebApp_results.xlsx"
            # -----------------------------------------------------------

            # db_record = self.gridfs.get(self.jobid)
            # buffer = db_record.read()
            # project_name = db_record.project_name
            # if project_name:
            #     filename = project_name.replace(" ", "_") + '_NTA_WebApp_results.xlsx'
            # else:
            #     filename = self.jobid + '_NTA_WebApp_results.xlsx'

            # excel_filename = self.parameters['project_name'][1] + '_' + self.jobid + '.xlsx'
            # zipf.writestr('summary.xlsx', excel_data)
            # filename = "temp_filename.xlsx"
            zipf.writestr(filename, excel_data)

            # self.add_tracer_plots_to_zip(zipf, self.jobid)
            try:
                self.add_tracer_plots_to_zip(zipf, self.jobid)
            except (OperationFailure, TypeError, NoFile) as e:
                pass  # do we want to do anything if no tracer file present?

            try:
                self.add_occurrence_heatmap_to_zip(zipf, self.jobid)
            except (OperationFailure, TypeError, NoFile) as e:
                pass  # do we want to do anything if no heatmap plot present?

            try:
                self.add_cv_scatterplot_to_zip(zipf, self.jobid)
            except (OperationFailure, TypeError, NoFile) as e:
                pass  # do we want to do anything if no cv_scatterplot present?

        zip_filename = "nta_results_" + self.jobid + ".zip"
        response = HttpResponse(in_memory_zip.getvalue(), content_type="application/zip")
        response["Content-Disposition"] = "attachment; filename=" + zip_filename
        response["Content-length"] = in_memory_zip.tell()
        return response

    def decision_tree(self):
        data_pos_id = self.jobid + "_" + "All Detection Statistics (Pos)"
        data_neg_id = self.jobid + "_" + "All Detection Statistics (Neg)"
        try:  #
            data_pos = self.gridfs.get(data_pos_id)
            json_string = data_pos.read().decode("utf-8")
            pos_df = pd.read_json(json_string, orient="split")
        except (OperationFailure, TypeError, NoFile):
            pos_df = pd.DataFrame()  # if pos file does not exist
        try:
            data_neg = self.gridfs.get(data_neg_id)
            json_string = data_neg.read().decode("utf-8")
            neg_df = pd.read_json(json_string, orient="split")
        except (OperationFailure, TypeError, NoFile):
            neg_df = pd.DataFrame()  # if neg file does not exist
        combined_df = pd.concat([pos_df, neg_df])  # combine pos and neg mode stats files
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=tree_data.csv"
        combined_df.to_csv(path_or_buf=response, index_label=False, index=False)  # write our csv to the response
        return response

    def decision_tree_parameters(self):
        data_analysis_parameters_id = self.jobid + "_" + "Analysis Parameters"
        try:  #
            data_analysis_parameters = self.gridfs.get(data_analysis_parameters_id)
            json_string = data_analysis_parameters.read().decode("utf-8")
            analysis_parameters_df = pd.read_json(json_string, orient="split")
        except (OperationFailure, TypeError, NoFile):
            analysis_parameters_df = pd.DataFrame()  # if pos file does not exist
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=tree_data.csv"
        analysis_parameters_df.to_csv(
            path_or_buf=response, index_label=False, index=False
        )  # write our csv to the response
        return response
