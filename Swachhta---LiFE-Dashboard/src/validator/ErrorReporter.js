import { errors } from "@vinejs/vine";
export default class ErrorReporter {
  hasErrors = false;
  errors = {};
  report(message, rule, field, meta) {
    this.hasErrors = true;
    this.errors[field.wildCardPath] = message;
  }
  createError() {
    return new errors.E_VALIDATION_ERROR(this.errors);
  }
}
